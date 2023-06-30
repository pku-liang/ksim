#ifndef KSIM_DIALECT
#define KSIM_DIALECT

include "mlir/IR/OpBase.td"

def KSimDialect : Dialect {
  let name = "ksim";
  let cppNamespace = "::ksim";
  let summary = "Types and operations for the KSim tool";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    /// Register all Seq types.
    void registerTypes();
  }];
}

#endif
