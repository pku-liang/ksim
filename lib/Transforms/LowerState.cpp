#include "circt/Dialect/HW/HWTypes.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/Utils/RegInfo.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/Namespace.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <system_error>
#include <utility>

#define GEN_PASS_DEF_LOWERSTATE
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm; 

struct ModuleExternPattern : OpRewritePattern<hw::HWModuleExternOp> {
  using OpRewritePattern<hw::HWModuleExternOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hw::HWModuleExternOp op, PatternRewriter & rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

struct TopoInfo {
  llvm::DenseMap<Operation*, SmallVector<Operation*>> extraEdges;
  void addTopoConstraint(Operation * in, Operation * out) {
    extraEdges[in].push_back(out);
  }
  struct CycleInfo {
    bool vst = false;
    bool instack = false;
    Operation * parent;
  };
  void findCycle(llvm::DenseMap<Operation *, CycleInfo> &info, Operation * op) {
    info[op].vst = true;
    info[op].instack = true;
    auto go = [&](Operation * v) {
      if(!info.count(v)) return;
      if(!info[v].vst) {
        info[v].parent = op;
        findCycle(info, v);
      }
      else if(info[v].instack) {
        errs() << "cycle found: \n";
        for(auto p = op; p != v; p = info[p].parent) {
          errs() << p << " " << p->getName() << " " << p->getResult(0) << "\n";
          p->emitError() << "cycle";
        }
        errs() << v << " " << v->getName() << " " << v->getResult(0) << "\n";
        v->emitError() << "cycle";
      }
    };
    for(auto v: extraEdges[op]) go(v);
    for(auto user: op->getUsers()) go(user);
    info[op].instack = false;
  }
  void findCycle(SmallVector<Operation*> & ops) {
    llvm::DenseMap<Operation *, CycleInfo> info;
    for(auto op: ops) info[op];
    for(auto op: ops) {
      if(!info[op].vst) findCycle(info, op);
    }
  }
  void topoSort(SmallVector<Operation*> & ops) {
    findCycle(ops);
    struct OpInfo {
      unsigned idx = 0;
      unsigned deg = 0;
    };
    llvm::DenseMap<Operation*, OpInfo> deg;
    unsigned idx = 0;
    for(auto op: ops) deg[op].idx = idx++;
    using qnode_t = std::pair<unsigned, Operation*>;
    std::priority_queue<qnode_t, std::vector<qnode_t>, std::greater<qnode_t>> worklist;
    for(auto &[u, egs]: extraEdges) {
      for(auto v: egs) {
        if(deg.count(v)) deg[v].deg++;
      }
    }
    for(auto op: ops) {
      for(auto in: op->getOperands()) {
        deg[op].deg += deg.count(in.getDefiningOp());
      }
      if(deg[op].deg == 0) worklist.push(std::make_pair(deg[op].idx, op));
    }
    for(auto &op: ops) {
      assert(!worklist.empty() && "loop in topo sort");
      op = worklist.top().second; worklist.pop();
      for(auto user: op->getUsers()) {
        if(deg.count(user) && --deg[user].deg == 0) {
          worklist.push(std::make_pair(deg[user].idx, user));
        }
      }
      for(auto v: extraEdges[op]) {
        if(deg.count(v) && --deg[v].deg == 0) {
          worklist.push(std::make_pair(deg[v].idx, v));
        }
      }
    }
  }
};

struct PrefixNamespace {
  StringRef prefix;
  Namespace ns;
  PrefixNamespace(StringRef prefix): prefix(prefix) {}
  inline StringRef newName(const llvm::Twine & name) {return ns.newName(prefix + name);}
};

template<typename T>
struct LowerStateRewritePattern: OpRewritePattern<T> {
  PrefixNamespace & ns;
  Block* topBlock;
  TopoInfo & topoInfo;
  LowerStateRewritePattern(PrefixNamespace & ns, Block * topBlock, TopoInfo & topoInfo, MLIRContext * ctx)
    : OpRewritePattern<T>(ctx, 1), ns(ns), topBlock(topBlock), topoInfo(topoInfo) {}
  void createQueue(mlir::Location loc, llvm::StringRef name, mlir::Type type, int64_t depth, ArrayRef<int64_t> delays, PatternRewriter & rewriter) const {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(topBlock);
    auto op = rewriter.create<ksim::DefQueueOp>(loc, name, type, depth, delays);
    op.setPrivate();
  }
  mlir::Value createPipe(StringRef qname, mlir::Value in, int64_t delay, PatternRewriter & rewriter) const {
    createQueue(in.getLoc(), qname, in.getType(), delay, delay, rewriter);
    auto qget = rewriter.create<ksim::GetQueueOp>(in.getLoc(), in.getType(), qname, delay - 1);
    auto qpush = rewriter.create<ksim::PushQueueOp>(in.getLoc(), qname, in);
    topoInfo.addTopoConstraint(qget, qpush);
    return qget.getResult();
  }
};

struct MemRewritePattern: LowerStateRewritePattern<ksim::MemOp> {
  using LowerStateRewritePattern<ksim::MemOp>::LowerStateRewritePattern;
  LogicalResult matchAndRewriteMemRead(StringRef memName, ksim::MemReadOp op, PatternRewriter & rewriter, SmallVector<Operation*> &readers) const {
    PatternRewriter::InsertionGuard guard(rewriter);
    auto delay = getFusedDelay(op).value_or(op.getLatency());
    ksim::LowReadMemOp readMemOp;
    if(delay) {
      auto addr = createPipe(ns.newName(memName + "__raddr"), op.getAddr(), delay, rewriter);
      auto en = createPipe(ns.newName(memName + "__ren"), op.getEn(), delay, rewriter);
      readMemOp = rewriter.create<ksim::LowReadMemOp>(op.getLoc(), op.getMemType().getElementType(), memName, addr, en);
    }
    else {
      readMemOp = rewriter.create<ksim::LowReadMemOp>(op.getLoc(), op.getMemType().getElementType(), memName, op.getAddr(), op.getEn());
    }
    readers.push_back(readMemOp);
    mlir::Value out = readMemOp.getResult();
    rewriter.replaceOp(op, out);
    return success();
  }
  LogicalResult matchAndRewriteMemWrite(StringRef memName, ksim::MemWriteOp op, PatternRewriter & rewriter, SmallVector<Operation*> &combWriters, SmallVector<Operation*> &seqWriters) const {
    PatternRewriter::InsertionGuard guard(rewriter);
    auto delay = getFusedDelay(op).value_or(op.getLatency());
    ksim::LowWriteMemOp writeMemOp;
    if(delay > 1) {
      auto addr = createPipe(ns.newName(memName + "__waddr"), op.getAddr(), delay - 1, rewriter);
      auto data = createPipe(ns.newName(memName + "__wdata"), op.getData(), delay - 1, rewriter);
      auto en = createPipe(ns.newName(memName + "__wen"), op.getEn(), delay - 1, rewriter);
      auto mask = createPipe(ns.newName(memName + "__wmask"), op.getMask(), delay - 1, rewriter);
      writeMemOp = rewriter.create<LowWriteMemOp>(op.getLoc(), memName, addr, data, en, mask, op.getMaskBits());
    }
    else {
      writeMemOp = rewriter.create<LowWriteMemOp>(op.getLoc(), memName, op.getAddr(), op.getData(), op.getEn(), op.getMask(), op.getMaskBits());
    }
    if(delay == 0) {
      combWriters.push_back(writeMemOp);
    }
    else {
      seqWriters.push_back(writeMemOp);
    }
    rewriter.eraseOp(op);
    return success();
  }
  LogicalResult matchAndRewrite(ksim::MemOp op, PatternRewriter & rewriter) const final {
    auto name = ns.newName(op.getSymName());
    rewriter.setInsertionPointToEnd(topBlock);
    rewriter.create<ksim::DefMemOp>(op.getLoc(), name, op.getMemType());
    rewriter.setInsertionPoint(op);
    SmallVector<Operation*> readers;
    SmallVector<Operation*> combWriters, seqWriters;
    for(auto writeHandle: op.getOperands()) {
      auto writeOp = writeHandle.getDefiningOp();
      if(auto memWriteOp = dyn_cast<ksim::MemWriteOp>(writeOp)) {
        if(failed(matchAndRewriteMemWrite(name, memWriteOp, rewriter, combWriters, seqWriters)))
          return failure();
      }
    }
    for(auto read: op->getUsers()) {
      if(auto memReadOp = dyn_cast<ksim::MemReadOp>(read)) {
        if(failed(matchAndRewriteMemRead(name, memReadOp, rewriter, readers)))
          return failure();
      }
    }
    for(auto w: combWriters) {
      for(auto r: readers) {
        topoInfo.addTopoConstraint(w, r);
      }
    }
    for(auto r: readers) {
      for(auto s: seqWriters) {
        topoInfo.addTopoConstraint(r, s);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct InstancePattern : LowerStateRewritePattern<hw::InstanceOp> {
  using LowerStateRewritePattern<hw::InstanceOp>::LowerStateRewritePattern;
  LogicalResult matchAndRewrite(hw::InstanceOp op, PatternRewriter & rewriter) const final {
    auto funcType = rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
    rewriter.setInsertionPointToStart(topBlock);
    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getInstanceName(), funcType);
    func.setPrivate();
    rewriter.setInsertionPoint(op);
    SmallVector<mlir::Value> opreands;
    for(auto [idx, operand]: llvm::enumerate(op->getOperands())) {
      auto piped = createPipe(ns.newName(op.getInstanceName() + std::to_string(idx)), operand, 1, rewriter);
      opreands.push_back(piped);
    }
    auto out = rewriter.create<func::CallOp>(op.getLoc(), func, opreands);
    rewriter.replaceOp(op, out.getResults());
    return success();
  }
};

struct QueueRewritePattern: LowerStateRewritePattern<ksim::QueueOp> {
  using LowerStateRewritePattern<ksim::QueueOp>::LowerStateRewritePattern;
  LogicalResult matchAndRewrite(ksim::QueueOp op, PatternRewriter & rewriter) const final {
    auto name = ns.newName(getSVNameHint(op).value_or("queue"));
    createQueue(op.getLoc(), name, op.getType(), op.getMaxDelay(), op.getDelay(), rewriter);
    rewriter.setInsertionPoint(op);
    auto pushQ = rewriter.create<ksim::PushQueueOp>(op.getLoc(), name, op.getInput());
    SmallVector<Value> retValues;
    retValues.reserve(op->getNumResults());
    for(auto [v, d]: llvm::zip(op.getResults(), op.getDelay())) {
      auto getQ = rewriter.create<ksim::GetQueueOp>(v.getLoc(), op.getType(), name, d - 1);
      retValues.push_back(getQ.getResult());
      topoInfo.addTopoConstraint(getQ, pushQ);
    }
    rewriter.replaceOp(op, ValueRange(retValues));
    return success();
  }
};

template<typename T>
struct RegLikeRewritePattern: LowerStateRewritePattern<T> {
  using LowerStateRewritePattern<T>::LowerStateRewritePattern;
  LogicalResult rewriteRegLike(Operation * op, int64_t delay, PatternRewriter & rewriter) const {
    RegInfo info(op);
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value data = info.data;
    ksim::GetQueueOp getQueueOp = nullptr;
    StringRef queueName = "";
    if(info.en || delay) {
      queueName = this->ns.newName(info.name.value_or("_reg"));
      this->createQueue(op->getLoc(), queueName, info.type, 1, 0, rewriter);
      getQueueOp = rewriter.create<ksim::GetQueueOp>(op->getLoc(), info.type, queueName, 0);
      if(info.en) {
        data = rewriter.create<comb::MuxOp>(op->getLoc(), info.en, info.data, getQueueOp.getResult());
      }
    }
    if(info.reset) {
      data = rewriter.create<comb::MuxOp>(op->getLoc(), info.reset, info.resetValue, data).getResult();
    }
    if(info.en || delay) {
      auto pushQueueOp = rewriter.create<ksim::PushQueueOp>(op->getLoc(), queueName, data);
      this->topoInfo.addTopoConstraint(getQueueOp, pushQueueOp);
    }
    if(delay) {
      rewriter.replaceOp(op, ValueRange(getQueueOp.getResult()));
    }
    else {
      rewriter.replaceOp(op, data);
    }
    return success();
  }
};

struct FirRegRewritePattern: RegLikeRewritePattern<seq::FirRegOp> {
  using RegLikeRewritePattern<seq::FirRegOp>::RegLikeRewritePattern;
  LogicalResult matchAndRewrite(seq::FirRegOp op, PatternRewriter & rewriter) const final {
    return rewriteRegLike(op, getFusedDelay(op).value_or(1), rewriter);
  }
};

struct CompRegRewritePattern: RegLikeRewritePattern<seq::CompRegOp> {
  using RegLikeRewritePattern<seq::CompRegOp>::RegLikeRewritePattern;
  LogicalResult matchAndRewrite(seq::CompRegOp op, PatternRewriter & rewriter) const final {
    return rewriteRegLike(op, getFusedDelay(op).value_or(1), rewriter);
  }
};

struct CompRegEnRewritePattern: RegLikeRewritePattern<seq::CompRegClockEnabledOp> {
  using RegLikeRewritePattern<seq::CompRegClockEnabledOp>::RegLikeRewritePattern;
  LogicalResult matchAndRewrite(seq::CompRegClockEnabledOp op, PatternRewriter & rewriter) const final {
    return rewriteRegLike(op, getFusedDelay(op).value_or(1), rewriter);
  }
};

static func::FuncOp rewriteModuleOp(hw::HWModuleOp mod, llvm::DenseMap<StringRef, StringRef> & nameMap, TopoInfo & topoInfo, llvm::SmallVector<Operation*> & outputSet) {
  OpBuilder builder(mod.getContext());
  builder.setInsertionPointToEnd(mod->getBlock());
  auto createQueue = [&](auto in) {
    auto [v, p] = in;
    auto op = builder.create<ksim::DefQueueOp>(v.getLoc(), nameMap.lookup(p.getName()), v.getType(), 1, 0);
    op.setPublic();
    return op;
  };
  auto outputOp = cast<hw::OutputOp>(mod.getBodyBlock()->getTerminator());
  auto inQueues = to_vector(map_range(llvm::zip(mod.getArguments(), mod.getPorts().inputs), createQueue));
  auto outGlobals = to_vector(map_range(llvm::zip(outputOp->getOperands(), mod.getPorts().outputs), createQueue));
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(mod.getBodyBlock());
    for(auto [v, q]: llvm::zip(mod.getArguments(), inQueues)) {
      v.replaceAllUsesWith(builder.create<ksim::GetQueueOp>(v.getLoc(), v.getType(), q.getSymName(), 0));
    }
    builder.setInsertionPointToEnd(mod.getBodyBlock());
    for(auto [v, q]: llvm::zip(outputOp->getOperands(), outGlobals)) {
      auto out = builder.create<ksim::PushQueueOp>(outputOp.getLoc(), q.getSymName(), v);
      outputSet.push_back(out);
    }
  }
  outputOp->erase();
  auto topName = nameMap.lookup(mod.getModuleName());
  auto funcOp =builder.create<func::FuncOp>(mod.getLoc(), topName, builder.getFunctionType({}, {}));
  for(auto attr: mod->getAttrs()) {
    if(attr.getName().getValue().startswith("ksim")){ 
      funcOp->setAttr(attr.getName().getValue(), attr.getValue());
    }
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    auto entry = funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    SmallVector<Operation *> ops;
    for(auto &op: *mod.getBodyBlock()) {
      ops.push_back(&op);
    }
    topoInfo.topoSort(ops);
    for(auto op: ops) {
      op->remove();
      builder.insert(op);
    }
    builder.create<func::ReturnOp>(mod.getLoc());
  }
  mod.erase();
  return funcOp;
}

static func::FuncOp emitCombEval(func::FuncOp func, llvm::SmallVector<Operation*> &outputOpSet) {
  IRMapping mapping;
  auto clone = func.clone(mapping);
  clone.setSymName((func.getSymName() + "__comb").str());
  llvm::DenseSet<Operation*> cloneOpSet;
  for(auto op: outputOpSet) {
    cloneOpSet.insert(mapping.lookup(op));
  }
  clone.walk([&](ksim::PushQueueOp op) {
    if(!cloneOpSet.contains(op))
      op->erase();
  });
  clone.walk([&](ksim::LowWriteMemOp op) {
    op->erase();
  });
  auto builder = OpBuilder(func);
  builder.setInsertionPointAfter(func);
  builder.insert(clone);
  return clone;
}

struct LowerStatePass : ksim::impl::LowerStateBase<LowerStatePass> {
  using ksim::impl::LowerStateBase<LowerStatePass>::LowerStateBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, comb::CombDialect>();
  }
  static StringRef getFitType(int64_t width) {
    if(width <= 8) return "uint8_t ";
    else if(width <= 16) return "uint16_t";
    else if(width <= 32) return "uint32_t";
    else if(width <= 64) return "uint64_t";
    else if(width <= 128) return "__uint128_t";
    else assert(false && "width too large");
  }
  static StringRef getDirectionStr(hw::PortDirection dir) {
    if(dir == hw::PortDirection::INOUT)   return "/* inout  */";
    if(dir == hw::PortDirection::INPUT)   return "/* input  */";
    if(dir == hw::PortDirection::OUTPUT)  return "/* output */";
    return "/* unknown */";
  }
  void runOnOperation() final {
    PrefixNamespace ns(prefix);
    llvm::DenseMap<StringRef, StringRef> nameMap;
    SmallVector<hw::PortInfo> portInfos;
    auto modlist = getOperation();
    for(auto mod: modlist.getOps<hw::HWModuleOp>()) {
      nameMap[mod.getModuleName()] = ns.newName(mod.getModuleName());
      for(auto port: mod.getAllPorts()) {
        auto name = port.getName();
        nameMap[name] = ns.newName(name);
        portInfos.push_back(port);
      }
    }
    auto & context = getContext();
    ConversionTarget target(context);
    target.addIllegalOp<ksim::MemOp>();
    target.addIllegalOp<ksim::MemReadOp>();
    target.addIllegalOp<ksim::MemWriteOp>();
    target.addIllegalOp<ksim::QueueOp>();
    target.addLegalOp<circt::ModuleOp>();
    target.addLegalDialect<comb::CombDialect>();
    target.addLegalDialect<ksim::KSimDialect>();
    target.addLegalDialect<sv::SVDialect>();
    target.addLegalDialect<hw::HWDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalOp<hw::InstanceOp>();
    target.addIllegalOp<hw::HWModuleExternOp>();
    RewritePatternSet patterns(&context);
    TopoInfo topoInfo;
    patterns.add<ModuleExternPattern>(&context);
    patterns.add<InstancePattern>(ns, modlist.getBody(), topoInfo, &context);
    patterns.add<MemRewritePattern>(ns, modlist.getBody(), topoInfo, &context);
    patterns.add<QueueRewritePattern>(ns, modlist.getBody(), topoInfo, &context);
    patterns.add<FirRegRewritePattern>(ns, modlist.getBody(), topoInfo, &context);
    patterns.add<CompRegRewritePattern>(ns, modlist.getBody(), topoInfo, &context);
    patterns.add<CompRegEnRewritePattern>(ns, modlist.getBody(), topoInfo, &context);
    if (failed(applyFullConversion(modlist, target, std::move(patterns))))
      return signalPassFailure();
    auto mods = to_vector(modlist.getOps<hw::HWModuleOp>());
    func::FuncOp topFunc = nullptr;
    llvm::SmallVector<Operation*> outputOpSet;
    for(auto mod: mods) {
      topFunc = rewriteModuleOp(mod, nameMap, topoInfo, outputOpSet);
    }
    if(topFunc) {
      std::optional<func::FuncOp> combEval = std::nullopt;
      if(this->emitComb) {
        combEval = emitCombEval(topFunc, outputOpSet);
      }
      emitDriver(topFunc, combEval, portInfos, nameMap);
    }
  }
  void emitDriver(func::FuncOp top, std::optional<func::FuncOp> combEval, SmallVector<hw::PortInfo> & portInfos, llvm::DenseMap<StringRef, StringRef> &nameMap) {
    int64_t outputAhead = 0, resetAhead = 0;
    if(top->hasAttr("ksim.output_ahead")) {
      outputAhead = top->getAttrOfType<IntegerAttr>("ksim.output_ahead").getInt();
    }
    if(top->hasAttr("ksim.reset_ahead")) {
      resetAhead = top->getAttrOfType<IntegerAttr>("ksim.reset_ahead").getInt();
    }
    if(!headerFile.empty()) {
      std::error_code ec;
      raw_fd_ostream header(headerFile, ec);
      assert(!ec);
      header << "#pragma once\n\n";
      header << "#ifdef __cplusplus\n";
      header << "#include<cstdint>\n";
      header << "extern \"C\"{\n";
      header << "#else\n";
      header << "#include<stdint.h>\n";
      header << "#endif\n";
      header << "\n\n";
      for(auto port: portInfos) {
        auto name = nameMap[port.getName()];
        if(name == "clock") continue;
        auto width = hw::getBitWidth(port.type);
        header << "extern " << getDirectionStr(port.direction);
        if(width <= 128) {
          header << " " << getFitType(width) << " " << name << "; // " << port.type << "\n";
        }
        else {
          auto bytes = (width + 7) / 8;
          header << " " << "uint8_t " << name << "[" << bytes << "]; //" << port.type << "\n";
        }
      }
      header << "\n\n";
      header << "void " << top.getSymName() << "();\n";
      header << "#define " << top.getSymName() << "_output_ahead " << outputAhead << "\n";
      header << "#define " << top.getSymName() << "_reset_ahead " << resetAhead << "\n";
      if(combEval.has_value()) {
        header << "void " << combEval->getSymName() << "();\n";
      }
      header << "\n\n";
      header << "#ifdef __cplusplus\n";
      header << "}\n";
      header << "#endif\n";
    }
    if(!driverFile.empty()) {
      std::error_code ec;
      raw_fd_ostream driver(driverFile, ec);
      assert(!ec);
      driver << "#include\"" << headerFile << "\"\n";
      driver << "#include<cstdlib>\n";
      driver << "#include<chrono>\n";
      driver << "#include<iostream>\n";
      driver << "\n";
      driver << "int main(int argc, char ** argv) {\n";
      driver << "  int cnt = atoi(argv[1]);\n";
      driver << "  reset = 1;\n";
      driver << "  for(auto i = " << top.getSymName() << "_reset_ahead; i >= 0; i--) {\n";
      driver << "    " << top.getSymName() << "();\n";
      driver << "    reset = 0;\n";
      driver << "  }\n";
      driver << "  auto start = std::chrono::system_clock::now();\n";
      driver << "  for(auto i = 0; i < cnt; i++) {\n";
      for(auto port: portInfos) {
        auto name = nameMap[port.getName()];
        if(name.contains("reset")) {
          driver << "    " << name << " = 0;\n";
        }
        else if(name != "clock") {
          auto width = hw::getBitWidth(port.type);
          if(width < 64) {
            driver << "    " << name << " = rand() & ((1ll << " << width << ") - 1);\n";
          }
          else if(width == 64) {
            driver << "    " << name << " = rand();\n";
          }
        }
      }
      driver << "    " << top.getSymName() << "();\n";
      driver << "  }\n";
      driver << "  auto stop = std::chrono::system_clock::now();\n";
      driver << "  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;\n";
      driver << "  return 0;\n";
      driver << "}\n";
    }
  }
};

std::unique_ptr<mlir::Pass> ksim::createLowerStatePass(LowerStateOptions options) {
  return std::make_unique<LowerStatePass>(options); 
}
