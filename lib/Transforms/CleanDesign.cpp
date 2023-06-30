#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "ksim/KSimPasses.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ksim/Utils/RegInfo.h"

#define GEN_PASS_DEF_CLEANDESIGN
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct PlusArgReaderRewritePattern : OpRewritePattern<hw::InstanceOp> {
  const ModuleOp top;
  PlusArgReaderRewritePattern(const ModuleOp top, MLIRContext * ctx): OpRewritePattern<hw::InstanceOp>(ctx, 1), top(top) {}
  LogicalResult matchAndRewrite(hw::InstanceOp op, PatternRewriter & rewriter) const final {
    auto refModule = SymbolTable::lookupSymbolIn(top, op.getModuleName());
    if(!refModule) return failure();
    auto extOp = dyn_cast<hw::HWModuleExternOp>(refModule);
    if(!extOp || extOp.getVerilogModuleName() != "plusarg_reader") return failure();
    for(auto param: op.getParameters()) {
      auto decl = param.cast<hw::ParamDeclAttr>();
      if(decl.getName() == "FORMAT") {
        auto fmt = decl.getValue().cast<StringAttr>().getValue();
        auto name = fmt.split('=').first;
        op.setInstanceName(("plusarg_" + name).str());
      }
    }
    return failure();
  }
};
struct PlusArgReaderRemovePattern : OpRewritePattern<hw::InstanceOp> {
  const ModuleOp top;
  PlusArgReaderRemovePattern(const ModuleOp top, MLIRContext * ctx): OpRewritePattern<hw::InstanceOp>(ctx, 1), top(top) {}
  LogicalResult matchAndRewrite(hw::InstanceOp op, PatternRewriter & rewriter) const final {
    auto refModule = SymbolTable::lookupSymbolIn(top, op.getModuleName());
    if(!refModule) return failure();
    auto extOp = dyn_cast<hw::HWModuleExternOp>(refModule);
    if(!extOp || extOp.getVerilogModuleName() != "plusarg_reader") return failure();
    rewriter.setInsertionPoint(op);
    SmallVector<Value> replValue;
    replValue.reserve(op.getNumResults());
    for(auto result: op->getResults()) {
      replValue.push_back(rewriter.create<hw::ConstantOp>(op.getLoc(), result.getType(), -1));
    }
    rewriter.replaceOp(op, replValue);
    return failure();
  }
};

static void removeEICG(circt::ModuleOp modlist) {
  SmallVector<Operation*> opToDelete;
  for(auto op: modlist.getOps<hw::HWModuleExternOp>()) {
    if(op.getVerilogModuleName() == "EICG_wrapper") {
      opToDelete.push_back(op);
    }
  }
  for(auto op: opToDelete) {
    op->erase();
  }
}
struct EICGRemovePattern: OpRewritePattern<hw::InstanceOp> {
  const ModuleOp top;
  EICGRemovePattern(const ModuleOp top, MLIRContext * ctx): OpRewritePattern<hw::InstanceOp>(ctx, 1), top(top) {}
  LogicalResult matchAndRewrite(hw::InstanceOp op, PatternRewriter & rewriter) const final {
    auto refModule = SymbolTable::lookupSymbolIn(top, op.getModuleName());
    if(!refModule) return failure();
    auto extOp = dyn_cast<hw::HWModuleExternOp>(refModule);
    if(!extOp || extOp.getVerilogModuleName() != "EICG_wrapper") return failure();
    rewriter.setInsertionPoint(op);
    mlir::Value clock = nullptr, en = nullptr;
    for(auto [port, operand]: llvm::zip(extOp.getPorts().inputs, op.getOperands())) {
      if(port.getName() == "test_en") {
        assert(cast<hw::ConstantOp>(operand.getDefiningOp()).getValue() == 0);
      }
      else if(port.getName() == "en") {
        en = operand;
      }
      else if(port.getName() == "in") {
        clock = operand;
      }
    }
    assert(en && "enable signal not found");
    assert(clock && "clock signal not found");
    auto gatedClock = op->getResult(0);
    SmallVector<std::pair<Operation*, mlir::Value>> repls;
    for(auto &use: gatedClock.getUses()) {
      auto user = use.getOwner();
      assert(isRegLike(user));
      RegInfo info(user);
      assert(info.clock == gatedClock);
      auto regEn = en;
      rewriter.setInsertionPoint(user);
      if(info.en) {
        regEn = rewriter.create<comb::AndOp>(user->getLoc(), en, info.en);
      }
      mlir::Value repl = nullptr;
      if(info.reset && info.resetValue) {
        repl = rewriter.create<seq::CompRegClockEnabledOp>(user->getLoc(), info.data, clock, regEn, info.reset, info.resetValue, info.name.value_or("_reg"));
      }
      else {
        repl = rewriter.create<seq::CompRegClockEnabledOp>(user->getLoc(), info.data, clock, regEn, info.name.value_or("_reg"));
      }
      repls.push_back({user, repl});
    }
    for(auto [op, repl]: repls) {
      rewriter.replaceOp(op, repl);
    }
    rewriter.eraseOp(op);
    return failure();
  }
};

struct CleanDesignPass : ksim::impl::CleanDesignBase<CleanDesignPass> {
  using ksim::impl::CleanDesignBase<CleanDesignPass>::CleanDesignBase;
  void runOnOperation() final {
    auto top = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<EICGRemovePattern>(top, &getContext());
    patterns.add<PlusArgReaderRemovePattern>(top, &getContext());
    if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
    removeEICG(top);
  }
};

std::unique_ptr<mlir::Pass> ksim::createCleanDesignPass() {
  return std::make_unique<CleanDesignPass>();
}
