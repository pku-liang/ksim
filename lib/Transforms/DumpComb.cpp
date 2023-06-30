#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/KSimPasses.h"
#include "ksim/Utils/RegInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <queue>
#include <system_error>

#define GEN_PASS_DEF_DUMPCOMB
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

SmallVector<int64_t> getOpTypeVec(Operation * op) {
  auto getAlignedWidth = [](int64_t width) {
    if(width == 1) return 1ll;
    auto align = 1ll<<std::min(std::max(Log2_64_Ceil(width), 3u), 8u);
    return (width + align - 1) / align * align;
  };
  auto getTypeWidth = [](Type tpe){
    if(auto memType = dyn_cast<MemType>(tpe)) {
      return hw::getBitWidth(memType.getElementType());
    }
    else if(auto arrType = dyn_cast<hw::ArrayType>(tpe)) {
      return hw::getBitWidth(arrType.getElementType());
    }
    else {
      return hw::getBitWidth(tpe);
    }
  };
  SmallVector<int64_t> result;
  result.reserve(op->getNumResults() + op->getNumOperands());
  for(auto tpe: op->getResultTypes()) {
    result.push_back(getAlignedWidth(getTypeWidth(tpe)));
  }
  for(auto tpe: op->getOperandTypes()) {
    result.push_back(getAlignedWidth(getTypeWidth(tpe)));
  }
  return result;
}

struct DumpCombPass : ksim::impl::DumpCombBase<DumpCombPass> {
  using ksim::impl::DumpCombBase<DumpCombPass>::DumpCombBase;
  void runOnOperation() final {
    std::error_code ec;
    raw_fd_ostream fout(file, ec);
    if(ec) return signalPassFailure();
    auto mod = getOperation();
    fout << "id,name,type,operands\n";
    mod->walk([&](Operation * op) {
      if(isa<mlir::ModuleOp>(op)) return;
      if(isa<hw::HWModuleLike>(op)) return;
      if(isa<hw::OutputOp>(op)) return;
      fout << op << "," << op->getName() << ",";
      auto typeVec = getOpTypeVec(op);
      llvm::interleave(typeVec, fout, "|");
      fout << ",";
      bool first = true;
      for(auto operand: op->getOperands()) {
        auto def = operand.getDefiningOp();
        if(!def) continue;
        if(!first) fout << "|";
        else first = false;
        fout << def;
      }
      fout << "\n";
    });
    fout.flush();
  }
};

std::unique_ptr<mlir::Pass> ksim::createDumpCombPass(DumpCombOptions options) {
  return std::make_unique<DumpCombPass>(options);
}