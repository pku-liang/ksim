#include "ksim/KSimPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include <optional>
#include <queue>
#include <fstream>
#include <system_error>
#include "ksim/KSimOps.h"
#include "mlir/Support/FileUtilities.h"
#define  GEN_PASS_DEF_PARTITION
#include "PassDetails.h"

using namespace ksim;
using namespace mlir;
using namespace circt;
using namespace llvm;

namespace {

static const char DefaultKaHyParConfig[] = {
  "# general\n"
  "mode=direct\n"
  "objective=km1\n"
  "seed=-1\n"
  "cmaxnet=1000\n"
  "vcycles=0\n"
  "# main -> preprocessing -> min hash sparsifier\n"
  "p-use-sparsifier=true\n"
  "p-sparsifier-min-median-he-size=28\n"
  "p-sparsifier-max-hyperedge-size=1200\n"
  "p-sparsifier-max-cluster-size=10\n"
  "p-sparsifier-min-cluster-size=2\n"
  "p-sparsifier-num-hash-func=5\n"
  "p-sparsifier-combined-num-hash-func=100\n"
  "# main -> preprocessing -> community detection\n"
  "p-detect-communities=true\n"
  "p-detect-communities-in-ip=true\n"
  "p-reuse-communities=false\n"
  "p-max-louvain-pass-iterations=100\n"
  "p-min-eps-improvement=0.0001\n"
  "p-louvain-edge-weight=hybrid\n"
  "# main -> coarsening\n"
  "c-type=ml_style\n"
  "c-s=1\n"
  "c-t=160\n"
  "# main -> coarsening -> rating\n"
  "c-rating-score=heavy_edge \n"
  "c-rating-use-communities=true\n"
  "c-rating-heavy_node_penalty=no_penalty\n"
  "c-rating-acceptance-criterion=best_prefer_unmatched\n"
  "c-fixed-vertex-acceptance-criterion=fixed_vertex_allowed\n"
  "# main -> initial partitioning\n"
  "i-mode=recursive\n"
  "i-technique=multi\n"
  "# initial partitioning -> coarsening\n"
  "i-c-type=ml_style\n"
  "i-c-s=1\n"
  "i-c-t=150\n"
  "# initial partitioning -> coarsening -> rating\n"
  "i-c-rating-score=heavy_edge \n"
  "i-c-rating-use-communities=true\n"
  "i-c-rating-heavy_node_penalty=no_penalty\n"
  "i-c-rating-acceptance-criterion=best_prefer_unmatched\n"
  "i-c-fixed-vertex-acceptance-criterion=fixed_vertex_allowed\n"
  "# initial partitioning -> initial partitioning\n"
  "i-algo=pool\n"
  "i-runs=20\n"
  "# initial partitioning -> bin packing\n"
  "i-bp-algorithm=worst_fit\n"
  "i-bp-heuristic-prepacking=false\n"
  "i-bp-early-restart=true\n"
  "i-bp-late-restart=true\n"
  "# initial partitioning -> local search\n"
  "i-r-type=twoway_fm\n"
  "i-r-runs=-1\n"
  "i-r-fm-stop=simple\n"
  "i-r-fm-stop-i=50\n"
  "# main -> local search\n"
  "r-type=kway_fm_hyperflow_cutter_km1\n"
  "r-runs=-1\n"
  "r-fm-stop=adaptive_opt\n"
  "r-fm-stop-alpha=1\n"
  "r-fm-stop-i=350\n"
  "# local_search -> flow scheduling and heuristics\n"
  "r-flow-execution-policy=exponential\n"
  "# local_search -> hyperflowcutter configuration\n"
  "r-hfc-size-constraint=mf-style\n"
  "r-hfc-scaling=16\n"
  "r-hfc-distance-based-piercing=true\n"
  "r-hfc-mbc=true\n"
};

struct MFFCExtractor {
  llvm::DenseMap<Operation*, size_t> deg;
  llvm::DenseMap<Operation*, size_t> curDeg;
  llvm::DenseMap<Operation*, size_t> mffcId;
  size_t nextMffcId = 0;
  MFFCExtractor(func::FuncOp op) {
    findMFFC(op);
  }
  llvm::SmallVector<Operation*> expandMFFC(Operation * op) {
    llvm::SmallVector<Operation*> result;
    std::queue<Operation*> q;
    q.push(op);
    size_t curMffcId = nextMffcId++;
    while(!q.empty()) {
      auto cur = q.front();
      q.pop();
      mffcId[cur] = curMffcId;
      result.push_back(cur);
      for(auto ope: cur->getOperands()) {
        if(auto opeOp = ope.getDefiningOp()) {
          if(curDeg.contains(opeOp)) {
            curDeg[opeOp] = deg[opeOp];
          }
          if(--curDeg[opeOp] == 0) {
            q.push(opeOp);
          }
        }
      }
    }
    curDeg.clear();
    return result;
  }
  void findMFFC(func::FuncOp container) {
    for(auto & opref: container.getOps()) {
      auto op = & opref;
      deg[op] = 0;
      for(auto ope: op->getOperands()) {
        if(auto opeOp = ope.getDefiningOp()) {
          deg[opeOp]++;
        }
      }
    }
    llvm::SmallVector<Operation*> q;
    for(auto [op, refCnt]: deg) {
      if(refCnt == 0) {
        q.push_back(op);
      }
    }
    while(!q.empty()) {
      llvm::SmallVector<Operation*> nextQ;
      for(auto op: q) {
        nextQ.append(expandMFFC(op));
      }
      for(auto op: nextQ) {
        deg[op] = 0;
      }
      q.clear();
      for(auto op: nextQ) {
        for(auto ope: op->getOperands()) {
          if(auto opOpe = ope.getDefiningOp()) {
            if(deg[opOpe] > 0 && --deg[opOpe] == 0) {
              q.push_back(opOpe);
            }
          }
        }
      }
    }
  }
};

static std::pair<llvm::DenseMap<Operation*, size_t>, size_t> extractMFFCMapping(func::FuncOp op) {
  MFFCExtractor extractor(op);
  return {extractor.mffcId, extractor.nextMffcId};
}

struct DepGraph {
  const llvm::DenseMap<Operation*, size_t> &mffcIdMapping;
  size_t mffcCnt;
  llvm::SmallVector<size_t> mffcSize;
  llvm::DenseSet<std::pair<size_t, size_t>> mffcEdges;
  llvm::SmallVector<llvm::SmallVector<size_t>> mffcFanin;
  llvm::SmallVector<llvm::SmallDenseSet<size_t>> propagateSet;
  DepGraph(const llvm::DenseMap<Operation*, size_t> &mffcIdMapping, size_t mffcCnt)
  : mffcIdMapping(mffcIdMapping),
    mffcCnt(mffcCnt),
    mffcSize(mffcCnt), 
    mffcFanin(mffcCnt),
    propagateSet(mffcCnt)
  {
    for(auto [op, id]: mffcIdMapping) {
      mffcSize[id]++;
      for(auto ope: op->getOperands()) {
        if(auto opeOp = ope.getDefiningOp()) {
          if(mffcIdMapping.contains(opeOp)) {
            auto from = mffcIdMapping.at(opeOp);
            if (mffcEdges.insert({from, id}).second) {
              // errs() << "adde " << from << " " << id << "\n";
              mffcFanin[id].push_back(from);
            }
          }
        }
      }
    }
  }
  void propagate() {
    llvm::SmallVector<size_t> deg(mffcCnt);
    for(size_t i = 0; i < mffcCnt; i++) {
      for(auto in: mffcFanin[i]) {
        deg[in]++;
      }
    }
    std::queue<size_t> q;
    for(size_t i = 0; i < mffcCnt; i++) {
      if(deg[i] == 0) {
        q.push(i);
      }
    }
    while(!q.empty()) {
      auto cur = q.front();
      q.pop();
      for(auto to: mffcFanin[cur]) {
        if(--deg[to] == 0) {
          q.push(to);
          propagateSet[to].insert(propagateSet[cur].begin(), propagateSet[cur].end());
        }
      }
    }
  }
};

struct StateInfo {
  size_t id;
  StringRef name;
  Operation * defOps=nullptr;
  llvm::SmallVector<Operation*> writeOps={};
  llvm::SmallVector<Operation*> readOps ={};
  size_t partId;
};

llvm::DenseMap<StringRef, StateInfo> extractStates(Operation * op) {
  llvm::DenseMap<StringRef, StateInfo> stateInfo;
  op->walk([&](Operation * walkOp) {
    llvm::TypeSwitch<Operation*, void>(walkOp)
    .Case<ksim::DefQueueOp>   ([&](auto op) {
      stateInfo[op.getSymName()].name   = op.getSymName();
      stateInfo[op.getSymName()].defOps = op;
    })
    .Case<ksim::DefMemOp>     ([&](auto op) {
      stateInfo[op.getSymName()].name   = op.getSymName();
      stateInfo[op.getSymName()].defOps = op;
    })
    .Case<ksim::PushQueueOp>  ([&](auto op) {stateInfo[op.getQueue()].writeOps.push_back(op);})
    .Case<ksim::PushQueueEnOp>([&](auto op) {stateInfo[op.getQueue()].writeOps.push_back(op);})
    .Case<ksim::GetQueueOp>   ([&](auto op) {stateInfo[op.getQueue()].readOps.push_back(op);})
    .Case<ksim::LowWriteMemOp>([&](auto op) {stateInfo[op.getMem()].writeOps.push_back(op);})
    .Case<ksim::LowReadMemOp> ([&](auto op) {stateInfo[op.getMem()].readOps.push_back(op);})
    .Default([&](auto){});
  });
  size_t nextStateId = 0;
  for(auto & pair: stateInfo) {
    pair.second.id = nextStateId++;
  }
  return stateInfo;
}

static llvm::SmallVector<Operation*> recursiveDuplicate(llvm::SmallVector<Operation*> seeds, const llvm::DenseMap<Operation*, size_t> & opOrder) {
  llvm::DenseSet<Operation*> visited;
  std::queue<Operation*> q;
  for(auto seed: seeds) {
    q.push(seed);
    visited.insert(seed);
  }
  while(!q.empty()) {
    auto cur = q.front();
    q.pop();
    for(auto op: cur->getOperands()) {
      if(auto opeOp = op.getDefiningOp()) {
        if(visited.insert(opeOp).second) {
          q.push(opeOp);
        }
      }
    }
  }
  llvm::SmallVector<Operation*> sorted(visited.begin(), visited.end());
  llvm::sort(sorted, [&](auto a, auto b) { return opOrder.at(a) < opOrder.at(b); });
  IRMapping mapping;
  llvm::SmallVector<Operation*> result;
  for(auto op: sorted) {
    auto cloned = op->clone();
    mapping.map(op->getResults(), cloned->getResults());
    result.push_back(cloned);
  }
  for(auto op: result) {
    for(auto & opOpe: op->getOpOperands()) {
      opOpe.set(mapping.lookupOrDefault(opOpe.get()));
    }
  }
  return result;
}

enum StateDirection {
  Read, Write, Def
};

struct StateOpInfo {
  StateDirection dir;
  StringRef name;
};

static std::optional<StateOpInfo> getStateName(Operation * op) {
  return llvm::TypeSwitch<Operation*, std::optional<StateOpInfo>>(op)
  .Case<ksim::DefQueueOp>   ([&](auto op) {return StateOpInfo{Def,   op.getSymName()};})
  .Case<ksim::DefMemOp>     ([&](auto op) {return StateOpInfo{Def,   op.getSymName()};})
  .Case<ksim::PushQueueOp>  ([&](auto op) {return StateOpInfo{Write, op.getQueue()};})
  .Case<ksim::PushQueueEnOp>([&](auto op) {return StateOpInfo{Write, op.getQueue()};})
  .Case<ksim::GetQueueOp>   ([&](auto op) {return StateOpInfo{Read,  op.getQueue()};})
  .Case<ksim::LowWriteMemOp>([&](auto op) {return StateOpInfo{Write, op.getMem()};})
  .Case<ksim::LowReadMemOp> ([&](auto op) {return StateOpInfo{Read,  op.getMem()};})
  .Default([&](auto){return std::nullopt;});
}

struct KaHyPar {
  size_t nNodes;
  llvm::SmallVector<size_t> nodeWeight;
  llvm::SmallVector<size_t> edgeWeight;
  llvm::SmallVector<llvm::SmallDenseSet<size_t>> edges;
  KaHyPar(size_t nNodes): nNodes(nNodes), nodeWeight(nNodes, 1) {}
  static std::string createTempFile(StringRef suffix) {
    SmallVector<char> pathvec;
    auto result = llvm::sys::fs::createTemporaryFile("rep", suffix, pathvec);
    assert(!result && "unable to create temp file");
    return pathvec.data();
  }
  void addEdge(size_t weight, llvm::SmallDenseSet<size_t> edge) {
    edgeWeight.push_back(weight);
    edges.push_back(std::move(edge));
  }
  void dump(StringRef filename) {
    std::error_code ec;
    raw_fd_ostream of(filename, ec);
    assert(!ec && "unable to dump graph file");
    of << edges.size() << " " << nNodes << " 11\n";
    for (auto [w, e]: zip(edgeWeight, edges)) {
      of << w;
      for(auto v: e) {
        of << " " << v + 1;
      }
      of << "\n";
    }
    for(auto n: nodeWeight) {
      of << n << "\n";
    }
  }
  void createConfigFile(StringRef filename) {
    std::error_code ec;
    raw_fd_ostream fs(filename, ec);
    assert(!ec && "unable to write configure file");
    fs << DefaultKaHyParConfig;
    fs.close();
  }
  llvm::SmallVector<size_t> partId;
  void parseResultFile(StringRef filename) {
    std::ifstream partFile(filename.str());
    partId.resize(nNodes);
    for(size_t i = 0; i < nNodes; i++) {
      if(!(partFile >> partId[i])) {
        partId[i] --;
        assert(false && "invalid result file");
      }
    }
    partFile.close();
  }
  void runPartition(size_t k, StringRef program="KaHyPar") {
    auto confFile = createTempFile("ini");
    createConfigFile(confFile);
    auto graphFile = createTempFile("hgr");
    dump(graphFile);
    auto k_str = std::to_string(k);
    auto eps_str = "0.03";
    auto seed_str = std::to_string(-1);
    SmallVector<StringRef> args = {
      program,
      "-h", graphFile,
      "-k", k_str,
      "-e", eps_str,
      "-o", "km1",
      "-m", "direct",
      "-p", confFile,
      "-w", "true",
    };
    auto logfile = createTempFile("log");
    SmallVector<std::optional<StringRef>> redirects = {std::nullopt, logfile, logfile};
    auto partFile = graphFile + ".part" + k_str + ".epsilon" + eps_str + ".seed" + seed_str + ".KaHyPar";
    auto programPath = llvm::sys::findProgramByName(program);
    assert(!!programPath && "can't found KaHyPar program");
    errs() << "KaHyPar cmdline: ";
    llvm::interleave(args, errs(), " ");
    errs() << "\n";
    auto retcode = llvm::sys::ExecuteAndWait(programPath->data(), args, std::nullopt, redirects);
    errs() << "KaHyPar log:\n";
    errs() << openInputFile(logfile)->getBuffer() << "\n";
    if(retcode || !llvm::sys::fs::exists(partFile)) {
      errs() << "\n";
      assert(false && "KaHyPar fail");
    }
    parseResultFile(partFile);
    llvm::sys::fs::remove(partFile);
    llvm::sys::fs::remove(logfile);
  }
};

struct PartitionInfo {
  size_t id;
  llvm::SmallVector<const StateInfo*,0> writeStates;
  llvm::DenseSet<StringRef> readNames;
  llvm::DenseSet<StringRef> writeNames;
  StringRef evalFuncName;
  StringRef updateFuncName;
  void buildGraph(OpBuilder & builder, const llvm::DenseMap<Operation*,size_t>& opOrder) {
    llvm::SmallVector<Operation*> writes;
    for(auto write: writeStates) {
      writeNames.insert(write->name);
      writes.append(write->writeOps.begin(), write->writeOps.end());
    }
    auto dup = recursiveDuplicate(writes, opOrder);
    for(auto op: dup) {
      auto info = getStateName(op);
      if(info && info->dir == Read) {
        readNames.insert(info->name);
      }
    }
    auto loc = builder.getUnknownLoc();
    auto functionType = builder.getFunctionType({}, {});
    auto evalFunc = builder.create<func::FuncOp>(loc, "partition_eval_" + std::to_string(id), functionType);
    auto updateFunc = builder.create<func::FuncOp>(loc, "partition_update_" + std::to_string(id), functionType);
    evalFuncName = evalFunc.getSymName();
    updateFuncName = updateFunc.getSymName();
    OpBuilder evalBuilder(builder.getContext());
    OpBuilder updateBuilder(builder.getContext());
    evalBuilder.setInsertionPointToEnd(evalFunc.addEntryBlock());
    updateBuilder.setInsertionPointToEnd(updateFunc.addEntryBlock());
    llvm::DenseMap<StringRef, size_t> usedNames;
    auto getNextName = [&](StringRef name) {
      return name + "_buf_" + std::to_string(usedNames[name]++);
    };
    for(auto op: dup) {
      auto info = getStateName(op);
      if(info && info->dir == Write) {
        for(auto [i, opOpe]: enumerate(op->getOpOperands())) {
          auto value = opOpe.get();
          auto name = builder.getStringAttr(getNextName(info->name));
          auto def = builder.create<ksim::DefQueueOp>(loc, name, value.getType(), 1, SmallVector<int64_t>());
          def->setAttr("partId", builder.getI64IntegerAttr(id));
          evalBuilder.create<ksim::PushQueueOp>(loc, name, value);
          auto newValue = updateBuilder.create<ksim::GetQueueOp>(loc, value.getType(), name, 0);
          opOpe.set(newValue);
        }
        updateBuilder.insert(op);
      } else {
        evalBuilder.insert(op);
      }
    }
    updateBuilder.create<func::ReturnOp>(loc);
    evalBuilder.create<func::ReturnOp>(loc);
  }
};

static void sortStates(ModuleOp mod) {
  llvm::SmallVector<Operation*> ops;
  llvm::DenseMap<Operation*, size_t> partOrder;
  llvm::DenseMap<Operation*, size_t> stateOrder;
  for(auto [i, opref]: enumerate(mod.getOps())) {
    auto state = getStateName(&opref);
    if(state && state->dir == Def) {
      ops.push_back(&opref);
      partOrder[&opref] = opref.getAttrOfType<IntegerAttr>("partId").getInt();
      stateOrder[&opref] = i;
    }
  }
  llvm::sort(ops, [&](Operation* a, Operation* b) {
    auto partA = partOrder[a];
    auto partB = partOrder[b];
    if(partA != partB) {
      return partA < partB;
    }
    return stateOrder[a] < stateOrder[b];
  });
  for(auto op: ops) {
    op->remove();
  }
  OpBuilder builder(mod->getContext());
  builder.setInsertionPointToStart(mod.getBody());
  for(auto op: ops) {
    builder.insert(op);
  }
}

struct PartitionPass : public ksim::impl::PartitionBase<PartitionPass> {
  using ksim::impl::PartitionBase<PartitionPass>::PartitionBase;
  void runOnOperation() {
    auto mod = getOperation();
    auto func = *mod.getOps<func::FuncOp>().begin();
    auto stateInfo = extractStates(mod);
    auto [mffcIdMapping, mffcCnt] = extractMFFCMapping(func);
    DepGraph dep(mffcIdMapping, mffcCnt);
    KaHyPar hgp(stateInfo.size());
    for(auto &[name, info]: stateInfo) {
      const auto id = info.id;
      hgp.nodeWeight[id] = std::max(info.writeOps.size(), 1ul);
      for(auto op: info.writeOps) {
        dep.propagateSet[mffcIdMapping[op]].insert(id);
      }
      for(auto op: info.readOps) {
        dep.propagateSet[mffcIdMapping[op]].insert(id);
      }
    }
    dep.propagate();
    for(size_t i = 0; i < dep.propagateSet.size(); i++) {
      hgp.addEdge(dep.mffcSize[i], std::move(dep.propagateSet[i]));
    }
    hgp.runPartition(components, kahypar);
    llvm::SmallVector<PartitionInfo> partitions(components);
    OpBuilder builder(&getContext());
    builder.setInsertionPointToEnd(mod.getBody());
    for(auto &[name, info]: stateInfo) {
      auto partId = info.partId = hgp.partId[info.id];
      info.defOps->setAttr("partId", builder.getI64IntegerAttr(partId));
      partitions[partId].writeStates.push_back(&info);
      partitions[partId].id = partId;
    }
    llvm::DenseMap<Operation*, size_t> opOrder;
    for(auto [id, op]: enumerate(func.getOps())) {
      opOrder[&op] = id;
    }
    for(auto &part: partitions) {
      part.buildGraph(builder, opOrder);
    }
    func.erase();
    sortStates(mod);
    if(!driverFile.empty()) {
      std::error_code ec;
      raw_fd_ostream output(driverFile, ec);
      output << "#include <cstdlib>\n";
      output << "#include <cstdlib>\n";
      output << "#include <omp.h>\n";
      output << "#include <chrono>\n";
      output << "#include <iostream>\n";
      output << "namespace chrono = std::chrono;\n";
      output << "\n";
      output << "extern \"C\" {\n";
      for(int i = 0; i < components; i++) {
        output << "    extern void partition_eval_" << i << "();\n";
        output << "    extern void partition_update_" << i << "();\n";
      }
      output << "}\n";
      output << "int main(int argc, char ** argv) {\n";
      output << "    omp_set_num_threads(" << components << ");\n";
      output << "    auto cnt = atoi(argv[1]);\n";
      output << "    auto start_point = chrono::system_clock::now();\n";
      output << "    for(auto i = 0; i < cnt; i++) {\n";
      output << "        #pragma omp parallel sections\n";
      output << "        {\n";
      for(int i = 0; i < components; i++) {
      output << "            #pragma omp section\n";
      output << "            partition_eval_" << i << "();\n";
      }
      output << "        }\n";
      output << "        #pragma omp parallel sections\n";
      output << "        {\n";
      for(int i = 0; i < components; i++) {
      output << "            #pragma omp section\n";
      output << "            partition_update_" << i << "();\n";
      }
      output << "        }\n";
      output << "    }\n";
      output << "    auto stop_point = chrono::system_clock::now();\n";
      output << "    std::cout << chrono::duration_cast<chrono::microseconds>(stop_point - start_point).count() << std::endl;\n";
      output << "    return 0;\n";
      output << "}\n";

      output.close();
    }
  }
};

}

std::unique_ptr<mlir::Pass> ksim::createPartitionPass(PartitionOptions options) {
  return std::make_unique<PartitionPass>(options);
}
