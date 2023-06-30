#include "circt/Support/Namespace.h"
#include "ksim/KSimPasses.h"
#include <memory>
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "ksim/Utils/RegInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#define GEN_PASS_DEF_ADDZERORESET
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct AddZeroResetPass : ksim::impl::AddZeroResetBase<AddZeroResetPass> {
  using ksim::impl::AddZeroResetBase<AddZeroResetPass>::AddZeroResetBase;
  void runOnOperation() final {
    auto mod = getOperation();
    mlir::Value reset = nullptr;
    Namespace ns;
    ns.newName("reset");
    for(auto port: mod.getAllPorts()) {
      ns.newName(port.getName());
    }
    for(auto [arg, port]: llvm::zip(mod.getArguments(), mod.getPorts().inputs)) {
      if(port.getName() == "reset") {
        reset = arg;
      }
    }
    for(auto port: mod.getPorts().outputs) {
      if(port.getName() == "reset") {
        port.name = StringAttr::get(&getContext(), ns.newName(port.getName()));
        unsigned argNum = port.argNum;
        mod.modifyPorts({}, std::make_pair(argNum, port), {}, argNum);
      }
    }
    mlir::Value forceClock = nullptr;
    SmallVector<Operation*> regs;
    for(auto &opRef: mod.getOps()) {
      auto op = &opRef;
      if(isRegLike(op)) {
        RegInfo info(op);
        forceClock = info.clock;
        regs.push_back(op);
      }
    }
    OpBuilder builder(&getContext());
    if(!reset) {
      reset = mod.appendInput("reset", builder.getI1Type()).second;
    }
    for(auto op: regs) {
      if(!isRegLike(op)) continue;
      RegInfo info(op);
      if(info.reset && !isa<BlockArgument>(info.reset)) continue;
      builder.setInsertionPoint(op);
      auto resetValue = builder.create<hw::ConstantOp>(op->getLoc(), info.type, 0);
      llvm::TypeSwitch<Operation*, void>(op)
        .template Case<seq::FirRegOp>([&](seq::FirRegOp op){
          auto reg = builder.create<seq::FirRegOp>(op.getLoc(), op.getNext(), forceClock, op.getNameAttr(), reset, resetValue);
          op.getData().replaceAllUsesWith(reg);
          op->erase();
        })
        .template Case<seq::CompRegOp>([&](seq::CompRegOp op){
          auto reg = builder.create<seq::CompRegOp>(op.getLoc(), op.getInput(), forceClock, reset, resetValue, op.getSymName().value_or(StringRef()));
          op.getResult().replaceAllUsesWith(reg);
          op->erase();
        })
        .template Case<seq::CompRegClockEnabledOp>([&](seq::CompRegClockEnabledOp op){
          auto reg = builder.create<seq::CompRegClockEnabledOp>(op.getLoc(), op.getInput(), forceClock, op.getClockEnable(), reset, resetValue, op.getSymName().value_or(StringRef()));
          op.getResult().replaceAllUsesWith(reg);
          op->erase();
        })
        .Default([&](auto){assert(false);});
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createAddZeroResetPass() {
  return std::make_unique<AddZeroResetPass>();
}
