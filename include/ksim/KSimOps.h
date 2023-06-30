#pragma once

#include "ksim/KSimDialect.h"
#include "ksim/KSimTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"

#define GET_OP_CLASSES
#include "ksim/KSim.h.inc"