#ifndef KSIM_TYPES
#define KSIM_TYPES

include "ksim/KSimDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "circt/Dialect/HW/HWTypes.td"

class KSimType<string name, list<Trait> traits=[]> : TypeDef<KSimDialect, name, traits>;

def MemType : KSimType<"Mem"> {
  let summary = "simulated memory type";
  let parameters = (ins
    "uint64_t":$depth,
    "mlir::IntegerType":$element_type
  );
  let mnemonic = "mem";
  let assemblyFormat = "$element_type `[` $depth `]`";
  let extraClassDeclaration = [{
    uint64_t getAddrBits() {return llvm::Log2_64_Ceil(getDepth());}
    uint64_t getElementWidth() {return getElementType().getWidth();}
  }];
}

#endif
