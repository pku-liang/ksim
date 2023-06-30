#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/KSimPasses.h"
#include "ksim/Utils/RegInfo.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

#define GEN_PASS_DEF_LOADFIRRTL
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

enum PortTypeInfo {Read, Write, ReadWrite};
enum FieldInfo {Addr, Clk, En, RData, WData, WMask, WMode};
struct PortInfo {
  PortTypeInfo type;
  FieldInfo field;
  size_t id;
};
static PortInfo parsePortName(StringRef name) {
  PortInfo ret;
  if(name.consume_front("W")) ret.type = PortTypeInfo::Write;
  else if(name.consume_front("RW")) ret.type = PortTypeInfo::ReadWrite;
  else if(name.consume_front("R")) ret.type = PortTypeInfo::Read;
  else assert(false && "unknown port name");
  name.consumeInteger(10, ret.id);
  assert(name.consume_front("_"));
  if(name == "addr") ret.field = FieldInfo::Addr;
  else if(name == "clk") ret.field = FieldInfo::Clk;
  else if(name == "en") ret.field = FieldInfo::En;
  else if(name == "wmask") ret.field = FieldInfo::WMask;
  else if(name == "mask") ret.field = FieldInfo::WMask;
  else if(name == "wmode") ret.field = FieldInfo::WMode;
  else if(name == "data") {
    if(ret.type == Write) ret.field = FieldInfo::WData;
    else if(ret.type == Read) ret.field = FieldInfo::RData;
    else assert(false && "invalid data field");
  }
  else if(name == "wdata") ret.field = FieldInfo::WData;
  else if(name == "rdata") ret.field = FieldInfo::RData;
  else assert(false && "unknown port name");
  return ret;
}
struct MemPort {
  Value addr, clk, en, rdata, wdata, wmask, wmode;
  Value & select(FieldInfo info) {
    switch(info) {
      case Addr: return addr;
      case Clk: return clk;
      case En: return en;
      case RData: return rdata;
      case WData: return wdata;
      case WMask: return wmask;
      case WMode: return wmode;
      default: assert(false && "unknown info");
    }
  }
};
struct MemPorts {
  SmallVector<MemPort> readPorts;
  SmallVector<MemPort> writePorts;
  SmallVector<MemPort> readWritePorts;
  MemPort & select(PortTypeInfo type, size_t id) {
    switch(type) {
      case Read: return readPorts[id];
      case Write: return writePorts[id];
      case ReadWrite: return readWritePorts[id];
      default: assert(false && "unknown port type");
    }
  }
  Value & select(PortInfo info) {
    return select(info.type, info.id).select(info.field);
  }
};

struct FIRRTLMemorySummary {
  hw::HWModuleGeneratedOp gen;
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t width;
  size_t depth;
  size_t readLatency;
  size_t writeLatency;
  size_t maskGran;
  RUWAttr readUnderWrite;
  hw::WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
  SmallVector<PortInfo> inputPortInfo;
  SmallVector<PortInfo> outputPortInfo;
  FIRRTLMemorySummary(hw::HWModuleGeneratedOp gen): gen(gen) {
    numReadPorts      = gen->getAttr("numReadPorts").cast<IntegerAttr>().getUInt();
    numWritePorts     = gen->getAttr("numWritePorts").cast<IntegerAttr>().getUInt();
    numReadWritePorts = gen->getAttr("numReadWritePorts").cast<IntegerAttr>().getUInt();
    width             = gen->getAttr("width").cast<IntegerAttr>().getUInt();
    depth             = gen->getAttr("depth").cast<IntegerAttr>().getInt();
    readLatency       = gen->getAttr("readLatency").cast<IntegerAttr>().getUInt();
    writeLatency      = gen->getAttr("writeLatency").cast<IntegerAttr>().getUInt();
    maskGran          = gen->getAttr("maskGran").cast<IntegerAttr>().getUInt();
    readUnderWrite    = RUWAttr(gen->getAttr("readUnderWrite").cast<IntegerAttr>().getUInt());
    writeUnderWrite   = hw::WUW(gen->getAttr("writeUnderWrite").cast<IntegerAttr>().getInt());
    copy(map_range(gen->getAttr("writeClockIDs").cast<ArrayAttr>(), [](Attribute attr){
      return attr.cast<IntegerAttr>().getInt();
    }), std::back_inserter(writeClockIDs));
    copy(map_range(gen.getPorts().inputs, [](hw::PortInfo info) {
      return parsePortName(info.getName());
    }), std::back_inserter(inputPortInfo));
    copy(map_range(gen.getPorts().outputs, [](hw::PortInfo info) {
      return parsePortName(info.getName());
    }), std::back_inserter(outputPortInfo));
  }
  void print(raw_ostream &os) {
    os << "nR:    " << numReadPorts << "\n";
    os << "nW:    " << numWritePorts << "\n";
    os << "nRW:   " << numReadWritePorts << "\n";
    os << "width: " << width << "\n";
    os << "depth: " << depth << "\n";
    os << "latR:  " << readLatency << "\n";
    os << "latW:  " << writeLatency << "\n";
    os << "Gran:  " << maskGran << "\n";
    os << "RUW:   " << readUnderWrite << "\n";
    os << "WUW:   " << writeUnderWrite << "\n";
  }
  inline friend raw_ostream &operator<<(raw_ostream &os, FIRRTLMemorySummary & summary) {
    summary.print(os);
    return os;
  }
  MemPorts parseInstanceOp(hw::InstanceOp op) {
    MemPorts ret;
    ret.readPorts.resize(numReadPorts);
    ret.writePorts.resize(numWritePorts);
    ret.readWritePorts.resize(numReadWritePorts);
    for(auto [portInfo, operand]: llvm::zip(inputPortInfo, op->getOperands())) {
      ret.select(portInfo) = operand;
    }
    for(auto [portInfo, result]: llvm::zip(outputPortInfo, op->getResults())) {
      ret.select(portInfo) = result;
    }
    return ret;
  }
};

struct FIRRTLMemoryPattern : OpRewritePattern<hw::InstanceOp> {
  const ModuleOp top;
  FIRRTLMemoryPattern(const ModuleOp top, MLIRContext * ctx): OpRewritePattern<hw::InstanceOp>(ctx, 1), top(top) {}
  LogicalResult matchAndRewrite(hw::InstanceOp op, PatternRewriter & rewriter) const final {
    auto refModule = SymbolTable::lookupSymbolIn(top, op.getModuleName());
    if(!refModule) return failure();
    auto generated = dyn_cast<hw::HWModuleGeneratedOp>(refModule);
    if(!generated) return failure();
    auto schema = cast<hw::HWGeneratorSchemaOp>(SymbolTable::lookupSymbolIn(top, generated.getGeneratorKind()));
    if(schema.getDescriptor() != "FIRRTL_Memory") return failure();
    FIRRTLMemorySummary summary(generated);
    auto ports = summary.parseInstanceOp(op);
    auto loc = op->getLoc();
    auto memOp = rewriter.create<MemOp>(loc, summary.depth, summary.width, op.getInstanceName());
    auto memType = memOp.getMemType();
    auto memHandle = memOp.getHandle();
    IRMapping valueMap;
    for(auto rport: ports.readPorts) {
      auto readOp = rewriter.create<MemReadOp>(loc, memHandle, rport.addr, rport.en, rport.clk, summary.readLatency);
      valueMap.map(rport.rdata, readOp.getResult());
    }
    auto one = rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
    for(auto wport: ports.writePorts) {
      if(wport.wmask == nullptr) {
        wport.wmask = one;
      }
      auto write = rewriter.create<MemWriteOp>(loc, memType, wport.addr, wport.wdata, wport.en, wport.clk, wport.wmask, summary.maskGran, summary.writeLatency);
      memOp.addMemWrite(write);
    }
    for(auto rwport: ports.readWritePorts) {
      if(rwport.wmask == nullptr) rwport.wmask = one;
      auto wmodeN = rewriter.create<comb::XorOp>(loc, one, rwport.wmode).getResult();
      auto readEn = rewriter.create<comb::AndOp>(loc, rwport.en, wmodeN).getResult();
      auto readOp = rewriter.create<MemReadOp>(loc, memHandle, rwport.addr, readEn, rwport.clk, summary.readLatency).getResult();
      auto writeEn = rewriter.create<comb::AndOp>(loc, rwport.en, rwport.wmode).getResult();
      auto write = rewriter.create<MemWriteOp>(loc, memType, rwport.addr, rwport.wdata, writeEn, rwport.clk, rwport.wmask, summary.maskGran, summary.writeLatency);
      memOp.addMemWrite(write);
      valueMap.map(rwport.rdata, readOp);
    }
    rewriter.replaceOp(op, to_vector(map_range(op->getResults(), [&](Value v){return valueMap.lookup(v);})));
    return success();
  }
};

static void removeFirrtlMemory(circt::ModuleOp modlist) {
  SmallVector<Operation*> opToDelete;
  for(auto generated: modlist.getOps<hw::HWModuleGeneratedOp>()) {
    auto schema = cast<hw::HWGeneratorSchemaOp>(SymbolTable::lookupSymbolIn(modlist, generated.getGeneratorKind()));
    if(schema.getDescriptor() == "FIRRTL_Memory") {
      opToDelete.push_back(generated);
    }
  }
  for(auto schema: modlist.getOps<hw::HWGeneratorSchemaOp>()) {
    if(schema.getDescriptor() == "FIRRTL_Memory") {
      opToDelete.push_back(schema);
    }
  }
  for(auto op: opToDelete) {
    op->erase();
  }
}

struct FIRRTLRegPattern : OpRewritePattern<seq::FirRegOp> {
  using OpRewritePattern<seq::FirRegOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(seq::FirRegOp op, PatternRewriter & rewriter) const final {
    auto next = op.getNext().getDefiningOp();
    if(!next) return failure();
    auto mux = dyn_cast<comb::MuxOp>(next);
    if(!mux) return failure();
    auto fval = mux.getFalseValue();
    if(fval.getDefiningOp() != op) return failure();
    std::string name;
    if(op.getInnerSym()) {
      name = *(op.getInnerSym());
    }
    else if(op->hasAttr("sv.namehint")) {
      name = op->getAttr("sv.namehint").cast<StringAttr>().str();
    }
    else {
      name = "_reg";
    }
    auto regEn = rewriter.create<seq::CompRegClockEnabledOp>(op.getLoc(), mux.getTrueValue(), op.getClk(), mux.getCond(), op.getReset(), op.getResetValue(), StringRef(name));
    rewriter.replaceOp(op, regEn->getResults());
    if(mux->getUsers().empty()) {
      rewriter.eraseOp(mux);
    }
    return success();
  }
};

static void inlineExternalModule(circt::ModuleOp modlist, hw::HWModuleOp mod) {
  auto context = mod->getContext();
  auto block = mod.getBodyBlock();
  SmallVector<hw::InstanceOp> instToDelete;
  SmallVector<std::pair<unsigned, hw::PortInfo>> insInputs;
  SmallVector<Value> inputValues;
  SmallVector<std::pair<unsigned, hw::PortInfo>> insOutputs;
  SmallVector<Value> outputValues;
  unsigned inIdx = mod.getNumInputs();
  unsigned outIdx = mod.getNumOutputs();
  Namespace ns;
  for(auto port: mod.getAllPorts()) {
    ns.newName(port.getName());
  }
  block->walk([&](hw::InstanceOp inst) {
    auto modlike = SymbolTable::lookupSymbolIn(modlist, inst.getModuleName());
    if(auto extop = dyn_cast<hw::HWModuleExternOp>(modlike)) {
      auto instName = inst.getInstanceName();
      auto ports = extop.getPorts();
      instToDelete.push_back(inst);
      for(auto [port, value]: llvm::zip(ports.inputs, inst.getOperands())) {
        hw::PortInfo info;
        info.name = StringAttr::get(context, ns.newName(instName + "_" + port.name.getValue()));
        info.direction = hw::PortDirection::OUTPUT;
        info.type = port.type;
        insOutputs.push_back(std::make_pair(outIdx, info));
        outputValues.push_back(value);
      }
      for(auto [port, value]: llvm::zip(ports.outputs, inst.getResults())) {
        hw::PortInfo info;
        info.name = StringAttr::get(context, ns.newName(instName + "_" + port.name.getValue()));
        info.direction = hw::PortDirection::INPUT;
        info.type = port.type;
        insInputs.push_back(std::make_pair(inIdx, info));
        inputValues.push_back(value);
      }
    }
  });
  hw::modifyModulePorts(mod, insInputs, insOutputs, {}, {}, block);
  auto output = cast<hw::OutputOp>(block->getTerminator());
  output->insertOperands(outIdx, outputValues);
  for(auto [idx, data]: llvm::enumerate(llvm::zip(insInputs, inputValues))) {
    auto [ins, val] = data;
    val.replaceAllUsesWith(block->getArgument(inIdx + idx));
  }
  for(auto inst: instToDelete) {
    inst->erase();
  }
}

struct LoadFIRRTLPass : ksim::impl::LoadFIRRTLBase<LoadFIRRTLPass> {
  using ksim::impl::LoadFIRRTLBase<LoadFIRRTLPass>::LoadFIRRTLBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<KSimDialect>();
  }
  void runOnOperation() final {
    auto top = getOperation();
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<FIRRTLMemoryPattern>(top, &getContext());
      patterns.add<FIRRTLRegPattern>(&getContext());
      if(failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
      removeFirrtlMemory(top);
    }
    auto mods = to_vector(top.getOps<hw::HWModuleOp>());
    for(auto mod: mods) {
      inlineExternalModule(top, mod);
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createLoadFIRRTLPass() {
  return std::make_unique<LoadFIRRTLPass>();
}
