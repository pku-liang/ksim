#include "circt/Dialect/HW/HWTypes.h"
#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/Utils/RegInfo.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DEF_TEMPORALFUSION
#include "PassDetails.h"

#include "lemon/core.h"
#include "lemon/list_graph.h"
#include "lemon/random.h"
#include "lemon/network_simplex.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <system_error>

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;
using namespace lemon;

using Random = lemon::Random;
using Graph = lemon::ListDigraph;
using Node = Graph::Node;
using Arc = Graph::Arc;
using NodeIt = Graph::NodeIt;
using ArcIt = Graph::ArcIt;
template<typename T> using NodeMap = Graph::NodeMap<T>;
template<typename T> using ArcMap = Graph::ArcMap<T>;

struct SeqGraph;

struct GraphSymbol {
  enum GraphSymbolKind {MemSymbolKind};
  const GraphSymbolKind kind;
  GraphSymbol(GraphSymbolKind kind): kind(kind) {}
  inline GraphSymbolKind getKind() const {return kind;}
  virtual ~GraphSymbol() = default;
};

struct OpPort: public SmallVector<std::optional<Node>> {
  using SmallVector<std::optional<Node>>::SmallVector;
  inline OpPort(size_t n): SmallVector(n, std::nullopt) {}
  void setAll(Node node) { for(auto & t: *this) t = node; }
  void clearOperand(Operation * op, OperandRange range) {
    auto begin = op->getOperands().begin();
    for(auto p = range.begin(); p != range.end(); p++) {
      (*this)[p - begin] = std::nullopt;
    }
  }
};

struct OpInOutPort {
  OpPort in, out;
  inline OpInOutPort() = default;
  inline OpInOutPort(Operation * op): in(op->getNumOperands()), out(op->getNumResults()) {}
  inline OpInOutPort(Operation * op, Node inNode, Node outNode): OpInOutPort(op) {
    in.setAll(inNode);
    out.setAll(outNode);
  }
};

struct GraphEntry {
  Operation * op;
  OpInOutPort port;
  GraphEntry(Operation * op, SeqGraph & g): op(op), port(op) {}
  virtual ~GraphEntry() = default;
  virtual int64_t computeCost(NodeMap<int64_t> & PMap) = 0;
  virtual void updateArcGrad(NodeMap<int64_t> & PMap, ArcMap<int64_t> & costMap, Random & rnd) = 0;
  virtual void rewrite(NodeMap<int64_t> &PMap) = 0;
  virtual void dumpDot(raw_ostream & out, SeqGraph & g) = 0;
  virtual int64_t getDumpStage(SeqGraph & g) = 0;
};

struct ValueTargetInfo {
  Arc arc;
  Node node;
  OpOperand * opOperand;
};

struct GraphValueEntry {
  Value value;
  Node source;
  GraphValueEntry(Value value, Node source, SeqGraph & g): value(value), source(source) {}
  virtual int64_t computeCost(NodeMap<int64_t> &PMap) = 0;
  virtual void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) = 0;
  virtual void rewrite(NodeMap<int64_t> &PMap, OpBuilder & builder, SeqGraph & g) = 0;
  virtual void dumpDot(raw_ostream & out, SeqGraph & g) = 0;
  std::optional<StringRef> name;
  void setName(StringRef name) {this->name=name;}
};

struct SeqGraph {
  Graph graph;

  ArcMap<int64_t> weightMap {graph};

  operator Graph& () {return graph;}
  inline Node addNode() {return graph.addNode();}
  inline Arc addEdge(Node from, Node to, int64_t weight) {
    Arc arc = graph.addArc(from, to);
    weightMap[arc] = weight;
    return arc;
  }
  inline int id(Node n) const {return graph.id(n);}

  llvm::DenseMap<Operation*, std::unique_ptr<GraphSymbol>> symEntries;

  template<typename T>
  T* getOrCreateSymbol(Operation * op) {
    if(symEntries.count(op)) return cast<T>(symEntries[op].get());
    auto & ent = symEntries[op] = std::make_unique<T>(op, *this);
    return cast<T>(ent.get());
  }

  Value clockSignal = nullptr;
  void setOrCheckClockSignal(Operation * op, Value clock) {
    if(clockSignal == nullptr) clockSignal = clock;
    else if(clockSignal != clock) op->emitError() << "all clock signal must be identical";
  }
  llvm::DenseMap<Operation*, std::unique_ptr<GraphEntry>> entries;
  llvm::DenseMap<Value, std::unique_ptr<GraphValueEntry>> valEntries;
  void build(hw::HWModuleOp op);
  void dumpDot(raw_ostream & out);

  NodeMap<int64_t> PMap {graph};
  void solve(int maxIter, bool verbose);

  int64_t computeCost(NodeMap<int64_t> &PMap);
  void computeSupplyMap(NodeMap<int64_t> &PMap, NodeMap<int64_t> &supplyMap, Random &rnd);

  llvm::DenseSet<Value> resetSignals;
  inline void markResetSignal(Value reset) {resetSignals.insert(reset);}

  void rewrite(MLIRContext * ctx);
};

struct MemSymbol : public GraphSymbol {
  ksim::MemOp op;
  Node memNode;
  static bool classof(const GraphSymbol *S) {return S->getKind() == MemSymbolKind;}
  MemSymbol(Operation * op, SeqGraph & g):
    GraphSymbol(MemSymbolKind), op(cast<MemOp>(op)), memNode(g.addNode()) { }
};

enum QueueType {
  NaiveQueueType,
  ShiftQueueType,
  PtrQueueType,
  VecQueueType
};

static int64_t computeQueueCost(int64_t width, int64_t depth) {
  return width * std::min(depth, 1l);
}

static int64_t computeQueueGrad(int64_t width, int64_t depth, Random & rnd) {
  static const auto mul = 1ll << 20;
  return mul * width / (1 + depth) + rnd.integer(128);
}

static void dumpDotNode(raw_ostream & out, int id, llvm::Twine label, int64_t P) {
  out << id << " [label=\"" << label << "," << P <<  "\"];\n";
}

struct RegEntry : public GraphEntry {
  RegInfo info;
  Node inNode, outNode;
  Arc costArc;
  int64_t dataWidth;
  RegEntry(Operation * op, SeqGraph & g): GraphEntry(op, g), info(op), inNode(g.addNode()), outNode(g.addNode()) {
    // g.addEdge(outNode, inNode, 0);
    costArc = g.addEdge(inNode, outNode, 1);
    port.in.setAll(inNode);
    port.out.setAll(outNode);
    dataWidth = hw::getBitWidth(info.data.getType());
    g.setOrCheckClockSignal(op, info.clock);
    g.markResetSignal(info.reset);
  }
  int64_t computeCost(NodeMap<int64_t> & PMap) final {
    auto depth = PMap[inNode] + 1 - PMap[outNode];
    if(info.en) {
      return computeQueueCost(dataWidth, 1) / 2 + computeQueueCost(dataWidth, depth) / 2;
    }
    else {
      return computeQueueCost(dataWidth, depth);
    }
  }
  void updateArcGrad(NodeMap<int64_t> & PMap, ArcMap<int64_t> & costMap, Random & rnd) final {
    auto depth = PMap[inNode] + 1 - PMap[outNode];
    auto grad = computeQueueGrad(dataWidth, depth, rnd);
    if(info.en) grad /= 2;
    costMap[costArc] = grad;
  }
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    out << "subgraph cluster_" << g.id(inNode) << "{\n";
    dumpDotNode(out, g.graph.id(inNode), op->getName().getStringRef() + "_in", g.PMap[inNode]);
    dumpDotNode(out, g.graph.id(outNode), op->getName().getStringRef() + "_out", g.PMap[outNode]);
    out << "}\n";
    out << g.graph.id(inNode) << " -> " << g.graph.id(outNode);
    if(g.PMap[inNode] + 1 - g.PMap[outNode]) out << "[color=blue]";
    out << ";\n";
  }
  int64_t getDumpStage(SeqGraph & g) final {return g.PMap[inNode];}
  void rewrite(NodeMap<int64_t> &PMap) final {
    setFusedDelay(op, PMap[inNode] + 1 - PMap[outNode]);
  }
};

struct CombEntry : public GraphEntry {
  Node node;
  CombEntry(Operation * op, SeqGraph & g): GraphEntry(op, g), node(g.addNode()) {
    port.in.setAll(node);
    port.out.setAll(node);
  }
  int64_t computeCost(NodeMap<int64_t> & PMap) final {return 0;}
  void updateArcGrad(NodeMap<int64_t> & PMap, ArcMap<int64_t> & costMap, Random & rnd) final {}
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpDotNode(out, g.graph.id(node), op->getName().getStringRef(), g.PMap[node]);
  }
  int64_t getDumpStage(SeqGraph & g) final {return g.PMap[node];}
  void rewrite(NodeMap<int64_t> &PMap) final {}
};

struct MemEntry : public GraphEntry {
  Node memNode;
  MemEntry(ksim::MemOp op, SeqGraph & g): GraphEntry(op, g) {
    memNode = g.getOrCreateSymbol<MemSymbol>(op)->memNode;
  }
  int64_t computeCost(NodeMap<int64_t> &PMap) final {return 0;}
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {}
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpDotNode(out, g.graph.id(memNode), op->getName().getStringRef(), g.PMap[memNode]);
  }
  int64_t getDumpStage(SeqGraph & g) final {return g.PMap[memNode];}
  void rewrite(NodeMap<int64_t> &PMap) final {}
};

struct MemReadEntry : public GraphEntry {
  Node inNode, memNode;
  Arc memArc;
  int64_t width, latency;
  ksim::MemReadOp getOp() {return cast<ksim::MemReadOp>(op);}
  MemReadEntry(ksim::MemReadOp op, SeqGraph & g): GraphEntry(op, g), inNode(g.addNode()) {
    auto memop = op.getMemOp();
    width = op.getMemType().getAddrBits() + 1;
    latency = op.getLatency();
    memNode = g.getOrCreateSymbol<MemSymbol>(memop)->memNode;
    memArc = g.addEdge(inNode, memNode, latency);
    port.in.setAll(inNode);
    port.in.clearOperand(op, op.getMemMutable());
    port.out.setAll(memNode);
    g.setOrCheckClockSignal(op, op.getClock());
  }
  inline int64_t computeDepth(NodeMap<int64_t> & PMap) {
    return PMap[inNode] + latency - PMap[memNode];
  }
  int64_t computeCost(NodeMap<int64_t> & PMap) final {
    auto depth = computeDepth(PMap);
    return computeQueueCost(width, depth);
  }
  void updateArcGrad(NodeMap<int64_t> & PMap, ArcMap<int64_t> & costMap, Random & rnd) final {
    auto depth = computeDepth(PMap);
    costMap[memArc] = computeQueueGrad(width, depth, rnd);
  }
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpDotNode(out, g.graph.id(inNode), op->getName().getStringRef() + "_in", g.PMap[inNode]);
    out << g.graph.id(inNode) << " -> " << g.graph.id(memNode) << ";\n";
  }
  int64_t getDumpStage(SeqGraph & g) final {return g.PMap[memNode];}
  void rewrite(NodeMap<int64_t> &PMap) final {
    setFusedDelay(op, computeDepth(PMap));
  }
};

struct MemWriteEntry : public GraphEntry {
  ksim::MemWriteOp getOp() {return cast<ksim::MemWriteOp>(op);}
  Node inNode, memNode;
  Arc memArc, revArc;
  int64_t addrWidth, dataWidth, latency;
  MemWriteEntry(ksim::MemWriteOp op, SeqGraph & g): GraphEntry(op, g), inNode(g.addNode()) {
    latency = op.getLatency();
    auto memop = op.getMemOp();
    memNode = g.getOrCreateSymbol<MemSymbol>(memop)->memNode;
    memArc = g.addEdge(inNode, memNode, latency);
    // revArc = g.addEdge(memNode, inNode, 0);
    port.in.setAll(inNode);
    addrWidth = hw::getBitWidth(op.getAddr().getType());
    dataWidth = hw::getBitWidth(op.getData().getType());
    g.setOrCheckClockSignal(op, op.getClock());
  }
  inline int64_t computeDepth(NodeMap<int64_t> & PMap) {
    return PMap[inNode] + latency - PMap[memNode];
  }
  int64_t computeCost(NodeMap<int64_t> & PMap) final {
    auto depth = computeDepth(PMap);
    return computeQueueCost(addrWidth, depth) 
      + computeQueueCost(dataWidth, depth) 
      + computeQueueCost(1, depth);
  }
  void updateArcGrad(NodeMap<int64_t> & PMap, ArcMap<int64_t> & costMap, Random & rnd) final {
    auto depth = computeDepth(PMap);
    costMap[memArc] = computeQueueGrad(addrWidth, depth, rnd)
      + computeQueueGrad(dataWidth, depth, rnd)
      + computeQueueGrad(1, depth, rnd);
  }
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpDotNode(out, g.graph.id(inNode), op->getName().getStringRef() + "_in", g.PMap[inNode]);
    out << g.graph.id(inNode) << " -> " << g.graph.id(memNode) << ";\n";
  }
  int64_t getDumpStage(SeqGraph & g) final {return g.PMap[memNode];}
  void rewrite(NodeMap<int64_t> &PMap) final {
    setFusedDelay(op, computeDepth(PMap));
  }
};

struct ConstantLikeEntry : public GraphEntry {
  ConstantLikeEntry(Operation * op, SeqGraph & g): GraphEntry(op, g) {}
  int64_t computeCost(NodeMap<int64_t> &PMap) final {return 0;}
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {}
  void dumpDot(raw_ostream & out, SeqGraph & g) final {}
  int64_t getDumpStage(SeqGraph & g) final {return 0;}
  void rewrite(NodeMap<int64_t> &PMap) final {}
};

using ValueTargetVec = SmallVector<ValueTargetInfo>;

static void dumpTargetEdges(ArrayRef<ValueTargetInfo> targets, Node source, raw_ostream & out, SeqGraph & g, std::optional<StringRef> name=std::nullopt) {
  auto psrc = g.PMap[source];
  if(name.has_value()) {
    dumpDotNode(out, g.graph.id(source), *name, g.PMap[source]);
  }
  for(auto [a, n, _]: targets) {
    auto ptarget = g.PMap[n];
    out << g.graph.id(source) << " -> " << g.graph.id(n) << "[";
    if(psrc - ptarget != 0) {
      out << "color=blue, label=\"" << psrc - ptarget << "\", constraint=false";
    }
    out << "]\n";
  }
}

static ValueTargetVec connectValueToTarget(mlir::Value value, Node valueNode, SeqGraph & g) {
  ValueTargetVec targets;
  for(auto &opOperand: value.getUses()) {
    auto user = opOperand.getOwner();
    auto idx = opOperand.getOperandNumber();
    auto toPort = g.entries[user]->port.in[idx];
    assert(toPort.has_value());
    auto toNode = toPort.value();
    auto arc = g.addEdge(valueNode, toNode, 0);
    targets.push_back({arc, toNode, &opOperand});
  }
  return targets;
}

static SmallVector<int64_t> computeTargetDelay(NodeMap<int64_t> &PMap, Node source, ValueTargetVec targets) {
  return to_vector(map_range(targets, [&](auto tgt){return PMap[source] - PMap[tgt.node];}));
}

static void rewriteTargets(mlir::Value clockSignal, mlir::Value value, ArrayRef<ValueTargetInfo> targets, ArrayRef<int64_t> delays, OpBuilder & builder, std::optional<StringRef> name=std::nullopt) {
  if(targets.empty()) return;
  if(*std::max_element(delays.begin(), delays.end()) == 0) return;
  auto delayVec = to_vector(delays);
  llvm::sort(delayVec);
  delayVec.erase(std::unique(delayVec.begin(), delayVec.end()), delayVec.end());
  if(delayVec.front() == 0) delayVec.erase(delayVec.begin());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(value);
  auto queueOp = builder.create<ksim::QueueOp>(value.getLoc(), value, clockSignal, delayVec);
  auto results = queueOp->getResults();
  for(auto [delay, tgt]: llvm::zip(delays, targets)) {
    if(delay != 0) {
      auto ptr = std::lower_bound(delayVec.begin(), delayVec.end(), delay) - delayVec.begin();
      tgt.opOperand->set(results[ptr]);
    }
  }
  if(name.has_value()) {
    setSVNameHint(queueOp, *name);
  }
  else if(auto defOp = value.getDefiningOp()) {
    auto namehint = getSVNameHint(defOp);
    if(namehint.has_value()) {
      setSVNameHint(queueOp, *namehint);
    }
  }
}

struct NaiveValueEntry : public GraphValueEntry {
  int64_t width;
  ValueTargetVec targets;
  NaiveValueEntry(Value value, Node source, SeqGraph & g)
    :GraphValueEntry(value, source, g), width(hw::getBitWidth(value.getType())) {
    targets = connectValueToTarget(value, source, g);
  }
  int64_t computeCost(NodeMap<int64_t> &PMap) final {
    auto delays = computeTargetDelay(PMap, source, targets);
    int64_t cost = 0;
    for(auto delay: delays) {
      cost += computeQueueCost(width, delay);
    }
    return cost;
  }
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {
    auto delays = computeTargetDelay(PMap, source, targets);
    for(auto [delay, info]: llvm::zip(delays, targets)) {
      costMap[info.arc] = computeQueueGrad(width, delay, rnd);
    }
  }
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpTargetEdges(targets, source, out, g, name);
  }
  void rewrite(NodeMap<int64_t> & PMap, OpBuilder & builder, SeqGraph & g) final {
    auto delays = computeTargetDelay(PMap, source, targets);
    rewriteTargets(g.clockSignal, value, targets, delays, builder, name);
  }
};

struct SimpleValueEntry : public GraphValueEntry {
  int64_t width;
  Node midNode;
  Arc midArc;
  ValueTargetVec targets;
  SimpleValueEntry(Value value, Node source, SeqGraph & g)
    : GraphValueEntry(value, source, g), width(hw::getBitWidth(value.getType())) {
    midNode = g.addNode();
    midArc = g.addEdge(source, midNode, 0);
    targets = connectValueToTarget(value, midNode, g);
  }
  int64_t computeCost(NodeMap<int64_t> &PMap) override {
    auto delays = computeTargetDelay(PMap, source, targets);
    auto mx = *std::max_element(delays.begin(), delays.end());
    return computeQueueCost(width, mx);
  }
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) override {
    auto midDelay = PMap[source] - PMap[midNode];
    auto delays = computeTargetDelay(PMap, source, targets);
    auto mx = *std::max_element(delays.begin(), delays.end());
    auto nxtCost = computeQueueGrad(width, mx - midDelay, rnd);
    auto midCost = computeQueueGrad(width, midDelay, rnd);
    costMap[midArc] = midCost;
    auto arcCost = nxtCost / targets.size();
    for(auto info: targets) costMap[info.arc] = arcCost + rnd.integer(16);
  }
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpTargetEdges(targets, source, out, g, name);
  }
  void rewrite(NodeMap<int64_t> & PMap, OpBuilder & builder, SeqGraph & g) final  {
    auto delays = computeTargetDelay(PMap, source, targets);
    rewriteTargets(g.clockSignal, value, targets, delays, builder, name);
  }
};

struct BlockArgEntry : public SimpleValueEntry {
  using SimpleValueEntry::SimpleValueEntry;
  int64_t computeCost(NodeMap<int64_t> &PMap) final {
    if(targets.empty()) return 0;
    auto delays = computeTargetDelay(PMap, midNode, targets);
    auto mx = *std::max_element(delays.begin(), delays.end());
    return computeQueueCost(width, mx);
  }
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {
    if(targets.empty()) return;
    auto delays = computeTargetDelay(PMap, midNode, targets);
    auto mx = *std::max_element(delays.begin(), delays.end());
    costMap[midArc] = rnd.integer(16);
    auto arcCost = mx / targets.size();
    for(auto info: targets) costMap[info.arc] = arcCost + rnd.integer(16);
  }
};

struct ZeroValueEntry : public GraphValueEntry {
  ValueTargetVec targets;
  ZeroValueEntry(Value value, Node source, SeqGraph & g)
    : GraphValueEntry(value, source, g) {
    targets = connectValueToTarget(value, source, g);
  }
  int64_t computeCost(NodeMap<int64_t> &PMap) final {return 0;}
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {}
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    dumpTargetEdges(targets, source, out, g, name);
  }
  void rewrite(NodeMap<int64_t> & PMap, OpBuilder & builder, SeqGraph & g) final {
    auto delays = computeTargetDelay(PMap, source, targets);
    rewriteTargets(g.clockSignal, value, targets, delays, builder, name);
  }
};

struct ModuleEntry : public GraphValueEntry {
  hw::HWModuleOp mod;
  std::optional<Node> rstNode;
  Node outputNode;
  ModuleEntry(Node source, hw::HWModuleOp mod, std::optional<Node> rstNode, Node outputNode, SeqGraph & g):
    GraphValueEntry(nullptr, source,  g), mod(mod), rstNode(rstNode), outputNode(outputNode) {
    // g.addEdge(outputNode, source, 0);
    // g.addEdge(source, outputNode, 0);
    // if(rstNode.has_value()) {
    //   g.addEdge(rstNode.value(), source, 0);
    // }
  }
  int64_t computeCost(NodeMap<int64_t> &PMap) final {return 0;}
  void updateArcGrad(NodeMap<int64_t> &PMap, ArcMap<int64_t> &costMap, Random &rnd) final {}
  void dumpDot(raw_ostream & out, SeqGraph & g) final {
    out << "label=\"module start: " << g.PMap[source] << "\";\n";
  }
  void rewrite(NodeMap<int64_t> &PMap, OpBuilder & builder, SeqGraph & g) final {
    mod->setAttr("ksim.output_ahead", builder.getI64IntegerAttr(PMap[outputNode] - PMap[source]));
    if(rstNode.has_value()) {
      mod->setAttr("ksim.reset_ahead", builder.getI64IntegerAttr(PMap[*rstNode] - PMap[source]));
    }
  }
};

static std::unique_ptr<GraphEntry> buildGraphEntry(Operation * op, SeqGraph & graph) {
  return llvm::TypeSwitch<Operation*, std::unique_ptr<GraphEntry>>(op)
    .template Case<seq::FirRegOp>([&](seq::FirRegOp op){return std::make_unique<RegEntry>(op, graph);})
    .template Case<seq::CompRegOp>([&](seq::CompRegOp op){return std::make_unique<RegEntry>(op, graph);})
    .template Case<seq::CompRegClockEnabledOp>([&](seq::CompRegClockEnabledOp op){return std::make_unique<RegEntry>(op, graph);})
    .template Case<ksim::MemOp>([&](ksim::MemOp op) {return std::make_unique<MemEntry>(op, graph);})
    .template Case<ksim::MemReadOp>([&](ksim::MemReadOp op){return std::make_unique<MemReadEntry>(op, graph);})
    .template Case<ksim::MemWriteOp>([&](ksim::MemWriteOp op){return std::make_unique<MemWriteEntry>(op, graph);})
    .template Case<hw::ConstantOp>([&](hw::ConstantOp op){return std::make_unique<ConstantLikeEntry>(op, graph);})
    .template Case<hw::AggregateConstantOp>([&](hw::AggregateConstantOp op){return std::make_unique<ConstantLikeEntry>(op, graph);})
    .template Case<sv::ConstantZOp>([&](sv::ConstantZOp op){return std::make_unique<ConstantLikeEntry>(op, graph);})
    .template Case<sv::ConstantXOp>([&](sv::ConstantXOp op){return std::make_unique<ConstantLikeEntry>(op, graph);})
    .Default([&](Operation * op){return std::make_unique<CombEntry>(op, graph);});
}

static std::unique_ptr<GraphValueEntry> buildGraphValueEntry(Value value, Node valueNode, SeqGraph & graph, bool isReset=false, bool isBlkArg=false) {
  auto targets = value.getUses();
  auto numTargets = std::distance(targets.begin(), targets.end());
  if(isReset) {
    return std::make_unique<ZeroValueEntry>(value, valueNode, graph);
  }
  else if(isBlkArg) {
    return std::make_unique<BlockArgEntry>(value, valueNode, graph);
  }
  else if(numTargets >= 2) {
    return std::make_unique<SimpleValueEntry>(value, valueNode, graph);
  }
  else {
    return std::make_unique<NaiveValueEntry>(value, valueNode, graph);
  }
}

void SeqGraph::build(hw::HWModuleOp op) {
  auto block = op.getBodyBlock();
  for(auto & operation: *block) {
    auto op = &operation;
    entries[op] = buildGraphEntry(op, *this);
  }
  std::optional<Node> rstNode;
  auto source = addNode();
  for(auto [arg, port]: llvm::zip(block->getArguments(), op.getPorts().inputs)) {
    if(arg == clockSignal) continue;
    bool isResetNode = resetSignals.contains(arg) && port.getName() == "reset";
    auto node = isResetNode ? addNode() : source;
    auto & entry = valEntries[arg] 
      = buildGraphValueEntry(arg, node, *this, isResetNode, true);
    entry->setName(port.getName());
    if(isResetNode) rstNode = node;
  }
  auto outputEntry = static_cast<CombEntry*>(entries[op.getBodyBlock()->getTerminator()].get());
  auto outputNode = outputEntry->node;
  auto & modEntry = valEntries[nullptr] = 
    std::make_unique<ModuleEntry>(source, op, rstNode, outputNode, *this);
  modEntry->setName("MODULE");
  for(auto & operation: *block) {
    auto op = &operation;
    for(auto [res, out]: llvm::zip(op->getResults(), entries[op]->port.out)) {
      if(out.has_value()) {
        valEntries[res] = buildGraphValueEntry(res, *out, *this, resetSignals.contains(res));
      }
    }
  }
}

void SeqGraph::dumpDot(raw_ostream & out) {
  out << "digraph {\n";
  out << "node[shape=record];\n";
  llvm::DenseMap<int64_t,SmallVector<GraphEntry*>> stageMap;
  int64_t mn_stage = 0;
  for(auto & [k, e]: entries) {
    auto stage = e->getDumpStage(*this);
    stageMap[stage].push_back(e.get());
    mn_stage = std::min(mn_stage, stage);
  }
  for(auto &[k, v]: stageMap) {
    // out << "subgraph cluster_stage_" << k - mn_stage << "{\n";
    for(auto e: v) {
      e->dumpDot(out, *this);
    }
    // out << "}\n";
  }
  for(auto & [k, v]: valEntries) {
    v->dumpDot(out, *this);
  }
  out << "}\n";
}

void SeqGraph::solve(int maxIter, bool verbose) {
  Random rnd {0};
  using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
  NodeMap<int64_t> PMap {graph};
  auto finalCost = computeCost(PMap);
  if(verbose) {
    errs() << "It: " << 0 << " Cost: " << finalCost << "\n";
  }
  for(int i = 1; i <= maxIter; i++) {
    Solver solver {graph};
    solver.costMap(weightMap);
    NodeMap<int64_t> supplyMap {graph};
    computeSupplyMap(PMap, supplyMap, rnd);
    solver.supplyMap(supplyMap);
    auto status = solver.run();
    assert(status == Solver::ProblemType::OPTIMAL);
    solver.potentialMap(PMap);
    auto curCost = computeCost(PMap);
    if(curCost <= finalCost) {
      solver.potentialMap(this->PMap);
      finalCost = curCost;
    }
    if(verbose) {
      errs() << "It: " << i << " Cost: " << curCost << "\n";
    }
  }
}

int64_t SeqGraph::computeCost(NodeMap<int64_t> &PMap) {
  int64_t cost = 0;
  for(auto &[_, e]: entries) {
    cost += e->computeCost(PMap);
  }
  for(auto &[_, e]: valEntries) {
    cost += e->computeCost(PMap);
  }
  return cost;
}

void SeqGraph::computeSupplyMap(NodeMap<int64_t> &PMap, NodeMap<int64_t> &supplyMap, Random &rnd) {
  ArcMap<int64_t> costMap {graph};
  for(auto &[_, e]: entries) {
    e->updateArcGrad(PMap, costMap, rnd);
  }
  for(auto &[_, e]: valEntries) {
    e->updateArcGrad(PMap, costMap, rnd);
  }
  for(ArcIt a(graph); a != INVALID; ++a) {
    auto cost = costMap[a];
    supplyMap[graph.source(a)] += cost;
    supplyMap[graph.target(a)] -= cost;
  }
}

void SeqGraph::rewrite(MLIRContext * ctx) {
  OpBuilder builder(ctx);
  for(auto &[_, e]: valEntries) {
    e->rewrite(PMap, builder, *this);
  }
  for(auto &[_, e]: entries) {
    e->rewrite(PMap);
  }
}

struct TemporalFusionPass : ksim::impl::TemporalFusionBase<TemporalFusionPass> {
  using ksim::impl::TemporalFusionBase<TemporalFusionPass>::TemporalFusionBase;
  bool canBeScheduledOn(hw::HWModuleOp op) {return op.isPublic();}
  void runOnOperation() {
    SeqGraph graph;
    graph.build(getOperation());
    if(!disableOptimization) {
      graph.solve(tol, verbose);
    }
    if(!graphOut.empty()) {
      std::error_code ec;
      raw_fd_ostream fout(graphOut, ec);
      graph.dumpDot(fout);
    }
    graph.rewrite(&getContext());
  }
};

std::unique_ptr<mlir::Pass> ksim::createTemporalFusionPass(TemporalFusionOptions options) {
  return std::make_unique<TemporalFusionPass>(options);
}
