#ifndef KSIM_OPS
#define KSIM_OPS

include "ksim/KSimDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "ksim/KSimTypes.td"

class KSimOp<string mnemonic, list<Trait> traits = []> :
    Op<KSimDialect, mnemonic, traits>;

def MemOp : KSimOp<"mem", [Symbol, Pure]> {
  let summary = "memory operation for simulation";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name, Variadic<MemType>:$write_handle);
  let results = (outs MemType:$handle);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "uint64_t":$depth, "mlir::IntegerType":$element_type, "mlir::StringRef":$sym_name, CArg<"mlir::ArrayRef<mlir::Value>","{}">:$write_handle), [{
      build($_builder, $_state, $_builder.getType<MemType>(depth, element_type), sym_name, write_handle);
    }]>,
    OpBuilder<(ins "uint64_t":$depth, "uint64_t":$element_width, "mlir::StringRef":$sym_name, CArg<"mlir::ArrayRef<mlir::Value>","{}">:$write_handle), [{
      build($_builder, $_state, depth, $_builder.getType<mlir::IntegerType>(element_width), sym_name, write_handle);
    }]>
  ];
  let extraClassDeclaration = [{
    MemType getMemType() {return getHandle().getType().cast<MemType>();}
    static mlir::StringRef getSymAttrStrName() {return "sym_name";}
    void addMemWrite(mlir::Value handle);
  }];
}

def MemReadOp : KSimOp<"read", [Pure]> {
  let summary = "simulated memory read";
  let description = summary;
  let arguments = (ins
    MemType:$mem,
    HWIntegerType:$addr,
    I1:$en,
    I1:$clock,
    UI64Attr:$latency
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$mem `[`  type($addr) $addr `]` `en` $en `clock` $clock `latency` $latency `:` type($mem) `->` type($result) attr-dict";
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$mem, "mlir::Value":$addr, "mlir::Value":$en, "mlir::Value":$clock, "uint64_t":$latency), [{
      build($_builder, $_state, mem.getType().cast<MemType>().getElementType(), mem, addr, en, clock, latency);
    }]>
  ];
  let extraClassDeclaration = [{
    MemOp getMemOp() {return llvm::cast<ksim::MemOp>(getMem().getDefiningOp());}
    MemType getMemType() {return getMem().getType().cast<MemType>();}
    mlir::IntegerType getAddrType() {return getAddr().getType().cast<mlir::IntegerType>();}
  }];
}

def MemWriteOp : KSimOp<"write", [Pure]> {
  let summary = "simulated memory write";
  let description = summary;
  let arguments = (ins
    HWIntegerType:$addr,
    AnyType:$data,
    I1:$en,
    I1:$clock,
    HWIntegerType:$mask,
    UI64Attr:$mask_bits,
    UI64Attr:$latency
  );
  let results = (outs MemType:$handle);
  let assemblyFormat = "`[` type($addr) $addr `]` `=` type($data) $data `en` $en `clock` $clock `mask` type($mask) $mask `per` $mask_bits `bits` `latency` $latency `:` type($handle) attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    MemOp getMemOp() {return llvm::cast<ksim::MemOp>(*getHandle().getUsers().begin());}
    MemType getMemType() {return getHandle().getType().cast<MemType>();}
    mlir::IntegerType getAddrType() {return getAddr().getType().cast<mlir::IntegerType>();}
    mlir::IntegerType getMaskType() {return getMask().getType().cast<mlir::IntegerType>();}
  }];
}

def QueueOp : KSimOp<"queue", [Pure]> {
  let summary = "a fixed latency shift register";
  let description = summary;
  let arguments = (ins AnyType:$input, I1:$clk, I64SmallVectorArrayAttr:$delay);
  let results = (outs Variadic<AnyType>:$results);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static llvm::StringRef getDelayAttrStrName() {return "delay";}
    mlir::Type getType() {return getInput().getType();}
    int64_t getMaxDelay() {return getDelay().back();}
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$clk, "mlir::ArrayRef<int64_t>":$delay)>
  ];
}

def DefQueueOp : KSimOp<"low.def_queue", [Symbol]> {
  let summary = "define queue";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type, I64Attr:$depth, I64SmallVectorArrayAttr:$delay);
  let assemblyFormat = "$sym_name `depth` $depth `:` $type `delay` $delay attr-dict";
  let builders = [
    OpBuilder<(ins "mlir::StringRef":$name, "mlir::Type":$type, "int64_t":$depth, CArg<"mlir::ArrayRef<int64_t>", "{}">:$delay), [{
      build($_builder, $_state, name, type, depth, ::llvm::SmallVector<int64_t, 8>(delay));
    }]>
  ];
}

def GetQueueOp : KSimOp<"low.get_queue", [Pure]> {
  let summary = "get first element in queue";
  let description = summary;
  let arguments = (ins FlatSymbolRefAttr:$queue, I64Attr:$idx);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$queue `[` $idx `]` `:` type($result) attr-dict";
}

def PushQueueOp : KSimOp<"low.push_queue"> {
  let summary = "push new element into queue";
  let description = summary;
  let arguments = (ins FlatSymbolRefAttr:$queue, AnyType:$input);
  let assemblyFormat = "$queue $input `:` type($input) attr-dict";
}

def PushQueueEnOp : KSimOp<"low.push_queue.en"> {
  let summary = "push new element into queue";
  let description = summary;
  let arguments = (ins FlatSymbolRefAttr:$queue, AnyType:$input, I1:$en);
  let assemblyFormat = "$queue $input `en` $en `:` type($input) attr-dict";
}

def DefMemOp : KSimOp<"low.def_mem", [Symbol]> {
  let summary = "def lowlevel memory";
  let description = summary;
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<MemType>:$type);
  let assemblyFormat = "$sym_name $type attr-dict";
}

def LowReadMemOp : KSimOp<"low.read_mem", [Pure]> {
  let summary = "read lowlevel memory op";
  let description = summary;
  let arguments = (ins FlatSymbolRefAttr:$mem, HWIntegerType:$addr, I1:$en);
  let results = (outs HWIntegerType:$result);
  let assemblyFormat = "$mem `[` $addr `]` `en` $en `:` type($addr) `->` type($result) attr-dict";
}

def LowWriteMemOp : KSimOp<"low.write_mem"> {
  let summary = "write lowlevel memory op";
  let description = summary;
  let arguments = (ins
    FlatSymbolRefAttr:$mem,
    HWIntegerType:$addr,
    HWIntegerType:$data,
    I1:$en,
    HWIntegerType:$mask,
    UI64Attr:$mask_bits
  );
  let assemblyFormat = "$mem `[` type($addr) $addr `]` `=` type($data) $data `with` `mask` type($mask) $mask `per` $mask_bits `bits` `en` $en attr-dict";
}

#endif
