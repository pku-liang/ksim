#include "circt/Dialect/HW/HWOps.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/KSimPasses.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <queue>

#define GEN_PASS_DEF_MILITOMOORE
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

bool isCombOp(Operation * op) {
  if(isa<KSimDialect>(op->getDialect())) {
    if(auto memReadOp = dyn_cast<ksim::MemReadOp>(op)) {
      return memReadOp.getLatency() == 0;
    }
    
  }
}

struct MiliToMoorePass : ksim::impl::MiliToMooreBase<MiliToMoorePass> {
  using ksim::impl::MiliToMooreBase<MiliToMoorePass>::MiliToMooreBase;
  void runOnOperation() final {
    auto mod = getOperation();
  }
};

std::unique_ptr<mlir::Pass> ksim::createMiliToMoorePass() {
  return std::make_unique<MiliToMoorePass>();
}