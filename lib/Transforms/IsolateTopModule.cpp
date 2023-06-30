#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InstanceGraphBase.h"

#include "llvm/ADT/DenseSet.h"

#define GEN_PASS_DEF_ISOLATETOPMODULE
#include "PassDetails.h"
#include "ksim/KSimPasses.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct IsolateTopModulePass: ksim::impl::IsolateTopModuleBase<IsolateTopModulePass> {
  using ksim::impl::IsolateTopModuleBase<IsolateTopModulePass>::IsolateTopModuleBase;
  void runOnOperation() final {
    auto modlist = getOperation();
    hw::HWModuleOp topModule = nullptr;
    modlist->walk([&](hw::HWModuleOp op) {
      bool isTop = topModuleName.empty() ? op.isPublic() : op.moduleName() == topModuleName;
      if(isTop) {topModule = op;}
      op.setPrivate();
    });
    if(!topModule) {
      modlist->emitError() << "can't find top module";
      return signalPassFailure();
    }
    topModule.setPublic();
    auto & instGraph = getAnalysis<hw::InstanceGraph>();
    llvm::DenseSet<hw::HWModuleLike> keepSet;
    std::function<void(hw::HWModuleLike)> findKeepSet = [&](hw::HWModuleLike op) {
      keepSet.insert(op);
      for(auto inst: *instGraph.lookup(op)) {
        auto target = inst->getTarget()->getModule();
        if(target) findKeepSet(target);
      }
    };
    findKeepSet(topModule);
    modlist->walk([&](hw::HWModuleLike modlike) {
      if(!keepSet.contains(modlike)) {
        modlike.erase();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> ksim::createIsolateTopModulePass(IsolateTopModuleOptions options) {
  return std::make_unique<IsolateTopModulePass>(options);
}