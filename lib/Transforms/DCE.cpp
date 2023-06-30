#include "circt/Dialect/HW/HWOps.h"
#include "ksim/KSimPasses.h"
#include "ksim/Utils/RegInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>
#include <queue>

#define GEN_PASS_DEF_DCE
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct DCEPass : ksim::impl::DCEBase<DCEPass> {
  using ksim::impl::DCEBase<DCEPass>::DCEBase;
  void deleteRecursively(llvm::DenseSet<Operation*> &toDelete, Operation * op) {
    toDelete.erase(op);
    for(auto user: op->getUsers()) {
      if(toDelete.contains(user))
        deleteRecursively(toDelete, user);
    }
    op->erase();
  }
  void runOnOperation() final {
    auto mod = getOperation();
    llvm::DenseSet<Operation*> keepOps;
    std::queue<Operation*> worklist;
    worklist.push(mod.getBodyBlock()->getTerminator());
    while(!worklist.empty()) {
      auto op = worklist.front(); worklist.pop();
      keepOps.insert(op);
      for(auto operand: op->getOperands()) {
        auto def = operand.getDefiningOp();
        if(def && !keepOps.contains(def)) {
          keepOps.insert(def);
          worklist.push(def);
        }
      }
    }
    llvm::DenseSet<Operation*> opToDelete;
    OpBuilder builder(&getContext());
    builder.setInsertionPointToEnd(mod.getBodyBlock());
    for(auto &op: mod.getOps()) {
      if(!keepOps.contains(&op)) {
        opToDelete.insert(&op);
      }
    }
    keepOps.clear();
    llvm::DenseSet<Operation*> constToDelete;
    for(auto op: opToDelete) {
      for(auto result: op->getResults()) {
        if(result.getType().isa<IntegerType>()) {
          auto repl = builder.create<hw::ConstantOp>(op->getLoc(), result.getType(), 0);
          result.replaceAllUsesWith(repl);
          constToDelete.insert(repl);
        }
      }
    }
    for(auto op: opToDelete) {
      deleteRecursively(opToDelete, op);
    }
    for(auto op: constToDelete) {
      op->erase();
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createDCEPass() {
  return std::make_unique<DCEPass>();
}
