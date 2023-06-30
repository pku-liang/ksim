#include "ksim/KSimTypes.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "ksim/KSimTypes.cpp.inc"

using namespace ksim;


void KSimDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ksim/KSimTypes.cpp.inc"
      >();
}
