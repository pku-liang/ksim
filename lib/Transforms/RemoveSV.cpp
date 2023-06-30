#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

#define GEN_PASS_DEF_REMOVESV
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct RemoveSVPass : ksim::impl::RemoveSVBase<RemoveSVPass> {
  using RemoveSVBase<RemoveSVPass>::RemoveSVBase;
  SmallVector<Operation*> svOps;
  static inline bool shouldRemove(Operation * op) {
    if(isa<sv::ConstantZOp>(op)) return false;
    if(isa<sv::ConstantXOp>(op)) return false;
    return op->getDialect()->getNamespace() == "sv";
  }
  void runOnOperation() final {
    auto mod = getOperation();
    for(auto & inner: mod.getOps()) {
      if(auto mod = dyn_cast<hw::HWModuleOp>(&inner)) {
        for(auto & inner: mod.getOps()) {
          if(shouldRemove(&inner)) {
            svOps.push_back(&inner);
          }
        }
      }
      else if(shouldRemove(&inner)) {
        svOps.push_back(&inner);
      }
    }
    for(auto op: llvm::reverse(svOps)) {
      op->erase();
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createRemoveSVPass() {
  return std::make_unique<RemoveSVPass>();
}