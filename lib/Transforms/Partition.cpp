#include "circt/Dialect/HW/HWOps.h"
#include "ksim/KSimPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
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

static std::pair<llvm::DenseMap<Operation*, size_t>, size_t> extractMFFC(func::FuncOp op) {
  MFFCExtractor extractor(op);
  return {extractor.mffcId, extractor.nextMffcId};
}

struct DepGraph {
  const llvm::DenseMap<Operation*, size_t> &mffcId;
  size_t mffcCnt;
  llvm::SmallVector<size_t> mffcSize;
  llvm::SmallVector<size_t> mffcExtraWeight;
  llvm::DenseSet<std::pair<size_t, size_t>> mffcEdges;
  llvm::SmallVector<llvm::SmallVector<size_t>> mffcFanin;
  llvm::SmallVector<llvm::DenseSet<size_t>> propagateSet;
  DepGraph(const llvm::DenseMap<Operation*, size_t> &mffcId, size_t mffcCnt): 
    mffcId(mffcId), mffcCnt(mffcCnt),
    mffcSize(mffcCnt), mffcExtraWeight(mffcCnt), mffcFanin(mffcCnt), propagateSet(mffcCnt)
  {
    for(auto [op, id]: mffcId) {
      mffcSize[id]++;
      for(auto ope: op->getOperands()) {
        if(auto opeOp = ope.getDefiningOp()) {
          if(mffcId.contains(opeOp)) {
            auto from = mffcId.at(opeOp);
            mffcEdges.insert({from, id});
            mffcFanin[id].push_back(from);

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
  void dumpHyperGraph(raw_fd_ostream & fout) {
    fout << mffcCnt << " " << propagateSet.size() << " 01\n";
    for(auto [id, ps]: enumerate(propagateSet)) {
      fout << mffcSize[id];
      for(auto p: ps) {
        fout << " " << mffcSize[p] + 1;
      }
      fout << "\n";
    }
  }
};

struct StateInfo {
  size_t id;
  StringRef name;
  Operation * defOpe=nullptr;
  llvm::SmallVector<Operation*> pushOpe={};
  llvm::SmallVector<Operation*> getOpe ={};
  size_t partId;
};

llvm::DenseMap<StringRef, StateInfo> stateAnalyze(Operation * op) {
  llvm::DenseMap<StringRef, StateInfo> stateInfo;
  op->walk([&](Operation * walkOp) {
    llvm::TypeSwitch<Operation*, void>(walkOp)
    .Case<ksim::DefQueueOp>   ([&](auto op) {
      stateInfo[op.getSymName()].name   = op.getSymName();
      stateInfo[op.getSymName()].defOpe = op;
    })
    .Case<ksim::DefMemOp>     ([&](auto op) {
      stateInfo[op.getSymName()].name   = op.getSymName();
      stateInfo[op.getSymName()].defOpe = op;
    })
    .Case<ksim::PushQueueOp>  ([&](auto op) {stateInfo[op.getQueue()].pushOpe.push_back(op);})
    .Case<ksim::PushQueueEnOp>([&](auto op) {stateInfo[op.getQueue()].pushOpe.push_back(op);})
    .Case<ksim::GetQueueOp>   ([&](auto op) {stateInfo[op.getQueue()].getOpe.push_back(op);})
    .Case<ksim::LowWriteMemOp>([&](auto op) {stateInfo[op.getMem()].pushOpe.push_back(op);})
    .Case<ksim::LowReadMemOp> ([&](auto op) {stateInfo[op.getMem()].getOpe.push_back(op);})
    .Default([&](auto){});
  });
  size_t nextStateId = 0;
  for(auto & pair: stateInfo) {
    pair.second.id = nextStateId++;
  }
  return stateInfo;
}

static std::string createTempFile(StringRef suffix) {
  SmallVector<char> pathvec;
  llvm::sys::fs::createTemporaryFile("rep", suffix, pathvec);
  return pathvec.data();
}

static std::string createConfFile() {
  auto configFile = createTempFile("ini");
  std::error_code ec;
  raw_fd_ostream fs(configFile, ec);
  assert(!ec && "unable to write configure file");
  fs << DefaultKaHyParConfig;
  fs.close();
  return configFile;
}

static std::string runKaHyPar(StringRef graph, StringRef program, size_t k) {
  auto conf = createConfFile();
  auto k_str = std::to_string(k);
  auto eps_str = "0.03";
  auto seed_str = std::to_string(-1);
  SmallVector<StringRef> args = {
    program,
    "-h", graph, "-k", k_str, "-e", eps_str,
    "-o", "km1", "-m", "direct", "-p", conf,
    "-w", "true",
  };
  auto logfile = createTempFile("log");
  SmallVector<std::optional<StringRef>> redirects = {std::nullopt, logfile, logfile};
  auto partFile = (graph + ".part" + k_str + ".epsilon" + eps_str + ".seed" + seed_str + ".KaHyPar").str();
  llvm::sys::fs::remove(partFile);
  auto programPath = llvm::sys::findProgramByName(program);
  assert(!!programPath && "can't found KaHyPar program");
  auto retcode = llvm::sys::ExecuteAndWait(programPath->data(), args, std::nullopt, redirects);
  if(retcode || !llvm::sys::fs::exists(partFile)) {
    errs() << "KaHyPar cmdline: ";
    llvm::interleave(args, errs(), " ");
    errs() << "\n";
    errs() << "KaHyPar log:\n";
    errs() << openInputFile(logfile)->getBuffer() << "\n";
    assert(false && "KaHyPar fail");
  }
  llvm::sys::fs::remove(logfile);
  return partFile;
}

static std::optional<llvm::SmallVector<size_t>> loadPartition(StringRef path, size_t N) {
  std::ifstream partFile(path.str());
  llvm::SmallVector<size_t> result(N);
  for(size_t i = 0; i < N; i++) {
    if(!(partFile >> result[i])) {
      return std::nullopt;
    }
  }
  return result;
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
      writes.append(write->pushOpe.begin(), write->pushOpe.end());
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
          auto def = builder.create<ksim::DefQueueOp>(loc, name, value.getType(), 1);
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
    auto stateInfo = stateAnalyze(mod);
    errs() << "state analysis\n";
    auto func = *mod.getOps<func::FuncOp>().begin();
    auto path = createTempFile("hgr");
    std::error_code ec;
    raw_fd_ostream fout(path, ec);
    if(ec) return signalPassFailure();
    auto [mffcId, mffcCnt] = extractMFFC(func);
    errs() << "mffc\n";
    DepGraph dep(mffcId, mffcCnt);
    for(auto &[name, info]: stateInfo) {
      const auto id = info.id;
      for(auto op: info.pushOpe) {
        dep.propagateSet[mffcId[op]].insert(id);
      }
      for(auto op: info.getOpe) {
        dep.propagateSet[mffcId[op]].insert(id);
      }
    }
    errs() << "extract dep\n";
    dep.propagate();
    dep.dumpHyperGraph(fout);
    fout.close();
    errs() << "dump hyper graph\n";
    errs() << "run KaHyPar\n";
    auto partFile = runKaHyPar(path, kahypar, components);
    errs() << "load configuration\n";
    auto resultOption = loadPartition(partFile, mffcCnt);
    if(resultOption->empty()) {
      errs() << "load partition failed\n";
      return signalPassFailure();
    }
    auto result = *resultOption;
    llvm::SmallVector<PartitionInfo> partitions(components);
    OpBuilder builder(&getContext());
    builder.setInsertionPointToEnd(mod.getBody());
    for(auto &[name, info]: stateInfo) {
      auto partId = info.partId = result[info.id];
      info.defOpe->setAttr("partId", builder.getI64IntegerAttr(partId));
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
    if(!hdrFile.empty()) {
      raw_fd_ostream header(hdrFile, ec);
      header << "#pragma once\n\n";
      header << "#ifdef __cplusplus\n";
      header << "#include<cstdlib>\n";
      header << "extern \"C\"{\n";
      header << "#else\n";
      header << "#include<stdlib.h>\n";
      header << "#endif\n";
      for(auto &part: partitions) {
        header << "void " << part.evalFuncName << "();\n";
        header << "void " << part.updateFuncName << "();\n";
      }
      header << "#ifdef __cplusplus\n";
      header << "}\n";
      header << "#endif\n";
      header << "const size_t numWorkers = " << components << ";\n";
      header << "void (*const f[][2])() = {\n";
      for(auto &part: partitions) {
        header << "{" << part.evalFuncName << ", " << part.updateFuncName << "}, \n";
      }
      header << "};\n";
      header.close();
    }
  }
};

}

std::unique_ptr<mlir::Pass> ksim::createPartitionPass(PartitionOptions options) {
  return std::make_unique<PartitionPass>(options);
}
