#pragma once

#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/TypeSwitch.h"

namespace ksim{

struct RegInfo {
  mlir::Value data = nullptr, reset = nullptr, resetValue = nullptr, clock = nullptr, out = nullptr, en = nullptr;
  mlir::Type type = nullptr;
  std::optional<mlir::StringRef> name;
  RegInfo(mlir::Operation * op) {
    using namespace circt;
    llvm::TypeSwitch<mlir::Operation*, void>(op)
      .template Case<seq::FirRegOp>([&](seq::FirRegOp op){
        data = op.getNext(); reset = op.getReset();
        resetValue = op.getResetValue();
        clock = op.getClk(); out = op.getResult();
        name = op.getName();
      })
      .template Case<seq::CompRegOp>([&](seq::CompRegOp op){
        data = op.getInput(); reset = op.getReset();
        resetValue = op.getResetValue();
        clock = op.getClk(); out = op.getResult();
        name = op.getSymName();
      })
      .template Case<seq::CompRegClockEnabledOp>([&](seq::CompRegClockEnabledOp op){
        data = op.getInput(); reset = op.getReset();
        resetValue = op.getResetValue();
        clock = op.getClk(); out = op.getResult();
        en = op.getClockEnable();
        name = op.getSymName();
      })
      .Default([&](auto op){assert(false);});
      type = data.getType();
  }
};

inline bool isRegLike(mlir::Operation * op) {
  using namespace circt;
  return llvm::TypeSwitch<mlir::Operation*, bool>(op)
    .template Case<seq::FirRegOp>([&](seq::FirRegOp op){
      return true;
    })
    .template Case<seq::CompRegOp>([&](seq::CompRegOp op){
      return true;
    })
    .template Case<seq::CompRegClockEnabledOp>([&](seq::CompRegClockEnabledOp op){
      return true;
    })
    .Default([&](auto op){return false;});
}

}