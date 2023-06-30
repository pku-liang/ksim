#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/KSimTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"

#include "ksim/KSimDialect.cpp.inc"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace ksim;

void KSimDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "ksim/KSim.cpp.inc"
  >();
}

LogicalResult MemWriteOp::verify() {
  if(getAddrType().getWidth() != std::max(getMemType().getAddrBits(), 1ul))
    return emitError() << "address width '" << getAddrType().getWidth() 
      << "' doesn't match memory address width '" << getMemType().getAddrBits();
  if(getMaskType().getWidth() * getMaskBits() != getMemType().getElementWidth())
    return emitError() << "invalid write mask, maskBits = " << getMaskType().getWidth() 
      << ", maskBits = " << getMaskBits() << ", elemWidth = " << getMemType().getElementWidth();
  return success();
}

LogicalResult MemReadOp::verify() {
  if(getAddrType().getWidth() != std::max(getMemType().getAddrBits(), 1ul))
    return emitError() << "address width '" << getAddrType().getWidth() 
      << "' doesn't match memory address width '" << getMemType().getAddrBits();
  return success();
}

LogicalResult QueueOp::verify() {
  int64_t last_delay = 0;
  for(auto d: getDelay()) {
    if(d < last_delay) {
      return emitError() << "delay must be ascending '" <<getDelay() << "'";
    }
    last_delay = d;
  }
  auto inputType = getInput().getType();
  if(getResults().size() != getDelay().size()) {
    return emitError() << "result size '" << getResults().size() << "' dosen't equal to latency size '" << getDelay().size() << "'";
  }
  if(!llvm::all_of(getResults(), [&](Value v){return v.getType() == inputType;})) {
    return emitError() << "input type '" << getInput().getType() << "' doesn't match out type '" << getResults().getType() << "'";
  }
  return success();
}

void QueueOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value input, Value clk, ArrayRef<int64_t> delay) {
  build(odsBuilder, odsState, TypeRange(SmallVector<Type>(delay.size(), input.getType())), input, clk, odsBuilder.getI64ArrayAttr(delay));
}

ParseResult QueueOp::parse(OpAsmParser & parser, OperationState & state) {
  auto & builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand input, clock;
  Type type;
  SmallVector<int64_t> delay;
  auto parseDelay = [&]() {
    int64_t x = 0;
    if(parser.parseInteger(x)) return failure();
    delay.push_back(x);
    return success();
  };
  if (parser.parseOperand(input) ||
      parser.parseKeyword(":") ||
      parser.parseType(type) ||
      parser.resolveOperand(input, type, state.operands) ||
      parser.parseKeyword("clock") ||
      parser.parseOperand(clock) ||
      parser.resolveOperand(clock, builder.getI1Type(), state.operands) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None, parseDelay) ||
      parser.parseOptionalAttrDict(state.attributes)) {
    return failure();
  }
  state.addAttribute(QueueOp::getDelayAttrStrName(), builder.getI64ArrayAttr(delay));
  return success();
}

void QueueOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getInput());
  p << " : ";
  p.printType(getInput().getType());
  p << " clock ";
  p.printOperand(getClk());
  p << " delay ";
  llvm::interleaveComma(getDelay(), p);
  p.printOptionalAttrDict((*this)->getAttrs(), QueueOp::getDelayAttrStrName());
}

ParseResult MemOp::parse(OpAsmParser & parser, OperationState & state) {
  ksim::MemType memType;
  mlir::StringAttr symName;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseSymbolName(symName) ||
      parser.parseCustomTypeWithFallback(memType) ||
      parser.parseKeyword("with") ||
      parser.parseKeyword("writer") ||
      parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.resolveOperands(operands, memType, state.operands))
    return failure();
  state.addAttribute(MemOp::getSymAttrStrName(), symName);
  state.types.reserve(1);
  state.types.push_back(memType);
  return success();
}

void MemOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  p.printStrippedAttrOrType(getMemType());
  p << " with writer ";
  p.printOperands(getOperands());
  p.printOptionalAttrDict((*this)->getAttrs(), MemOp::getSymAttrStrName());
}

void MemOp::addMemWrite(mlir::Value handle) {
  auto memType = handle.getType().cast<MemType>();
  assert(memType = getMemType());
  assert(isa<MemWriteOp>(handle.getDefiningOp()));
  getWriteHandleMutable().append(handle);
}

#define GET_OP_CLASSES
#include "ksim/KSim.cpp.inc"