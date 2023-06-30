#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

#define GEN_PASS_DEF_ADDLLVMDEBUGINFO
#include "PassDetails.h"

using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace ksim;

struct SVNameHintPattern: public RewritePattern {
  LLVM::DISubprogramAttr scope;
  SVNameHintPattern(LLVM::DISubprogramAttr scope, MLIRContext * context)
    : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context), scope(scope) {}
  LogicalResult matchAndRewrite(Operation * op, PatternRewriter & rewriter) const final {
    if(!op->hasAttr("sv.namehint")) return failure();
    auto loc = op->getLoc();
    auto lcloc = loc->findInstanceOf<mlir::FileLineColLoc>();
    if(!lcloc) return failure();
    auto val = op->getResult(0);
    auto one = rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getI32Type(), 1);
    auto ptrType = LLVM::LLVMPointerType::get(val.getType());
    rewriter.setInsertionPointToStart(op->getBlock());
    auto alloca = rewriter.create<LLVM::AllocaOp>(op->getLoc(), ptrType, one, 0);
    rewriter.setInsertionPointAfterValue(val);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), val, alloca);
    auto name = op->getAttrOfType<StringAttr>("sv.namehint").getValue();
    auto diType = LLVM::DIBasicTypeAttr::get(getContext(), 0x24, "unsigned int", hw::getBitWidth(val.getType()), 0x07);
    auto variableAttr = LLVM::DILocalVariableAttr::get(scope, name, scope.getFile(), lcloc.getLine(), 0, 0, diType);
    rewriter.create<LLVM::DbgDeclareOp>(op->getLoc(), alloca, variableAttr);
    op->removeAttr("sv.namehint"); // remove it to prevent endless loop
    return success();
  }
};

struct AddLLVMDebugInfoPass : ksim::impl::AddLLVMDebugInfoBase<AddLLVMDebugInfoPass> {
  void runOnOperation() {
    auto llvmFunc = getOperation();
    auto loc = llvmFunc.getLoc();
    auto fusedLoc = loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>();
    if(!fusedLoc) return;
    auto subprogram = fusedLoc.getMetadata();
    RewritePatternSet patterns(&getContext());
    patterns.add<SVNameHintPattern>(subprogram, &getContext());
    FrozenRewritePatternSet frozenPatternSet(std::move(patterns));
    if(failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatternSet))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createAddLLVMDebugInfoPass() {
  return std::make_unique<AddLLVMDebugInfoPass>();
}
