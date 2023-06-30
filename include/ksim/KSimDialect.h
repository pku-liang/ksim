#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "ksim/KSimDialect.h.inc"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include <optional>

namespace ksim {

inline mlir::StringRef getFusedDelayAttrName() {
  return "ksim.delay";
}

inline std::optional<int64_t> getFusedDelay(mlir::Operation * op) {
  if(op->hasAttr(getFusedDelayAttrName())) {
    return op->getAttrOfType<mlir::IntegerAttr>(getFusedDelayAttrName()).getInt();
  }
  else {
    return std::nullopt;
  }
}

inline void setFusedDelay(mlir::Operation * op, int64_t delay) {
  op->setAttr(getFusedDelayAttrName(), mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 64), mlir::APInt(64, delay)));
}

inline mlir::StringRef getSVNameHintAttrName() {
  return "sv.namehint";
}

inline std::optional<mlir::StringRef> getSVNameHint(mlir::Operation * op) {
  if(op->hasAttr(getSVNameHintAttrName())) {
    return op->getAttrOfType<mlir::StringAttr>(getSVNameHintAttrName()).getValue();
  }
  else {
    return std::nullopt;
  }
}

inline void setSVNameHint(mlir::Operation * op, mlir::StringRef name) {
  op->setAttr(getSVNameHintAttrName(), mlir::StringAttr::get(op->getContext(), name));
}

}