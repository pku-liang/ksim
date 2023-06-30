#ifndef KSIM_PASSES_TD
#define KSIM_PASSES_TD

include "mlir/Pass/PassBase.td"

def RemoveSV : Pass<"ksim-remove-sv", "circt::ModuleOp"> {
  let summary = "Remove SV Dialect";
  let description = summary;
  let constructor = "ksim::createRemoveSVPass()";
}

def IsolateTopModule : Pass<"ksim-isolate-top-module"> {
  let summary = "Isolate top module";
  let description = summary;
  let constructor = "ksim::createIsolateTopModulePass()";
  let options = [
    Option<"topModuleName", "top", "std::string", "", "The top module name">
  ];
}

def SlicePropagation : Pass<"ksim-slice-propagation", "circt::hw::HWModuleOp"> {
  let summary = "Slice propagation pass";
  let description = [{
    for operations like "slice(f(x))", if f is element wise operation, rewrite it to "f(slice(x))"
  }];
  let constructor = "ksim::createSlicePropagationPass()";
}

def AddZeroReset : Pass<"ksim-add-reset", "circt::hw::HWModuleOp"> {
  let summary = "Add zero reset to registers without reset value";
  let description = summary;
  let constructor = "ksim::createAddZeroResetPass()";
}

def Flatten : Pass<"ksim-flatten", "circt::ModuleOp"> {
  let summary = "Flatten all module";
  let description = "Flatten all module into top module.";
  let constructor = "ksim::createFlattenPass()";
  let statistics = [
    Statistic<"numOperations", "num-operations", "number of operations after flatten">
  ];
}

def DCE : Pass<"ksim-dce", "circt::hw::HWModuleOp"> {
  let summary = "dead code elimination";
  let description = summary;
  let constructor = "ksim::createDCEPass()";
}

def DumpComb : Pass<"ksim-dump-comb"> {
  let summary = "dump combnational logic graph to csv/dot";
  let description = summary;
  let constructor = "ksim::createDumpCombPass()";
  let options = [
    Option<"file", "file", "std::string", "", "dump file">
  ];
}

def MiliToMoore : Pass<"ksim-mili-to-moore", "circt::hw::HWModuleOp"> {
  let summary = "Convert small mili state machine to moore state machine";
  let description = [{
    Convert state machine like: [s', out] = f(s, in) into 
    $$
    \begin{aligned}
      s' = f_{s}(s, in)
      out = f_{out}(s, in)
    \end{aligned}
    $$
  }];
  let constructor = "ksim::createMiliToMoorePass()";
}

def LoadFIRRTL : Pass<"ksim-load-firrtl", "circt::ModuleOp"> {
  let summary = "Load firrtl object to ksim ir";
  let description = [{
    This pass will load firrtl compiler generated module,
    user defined FIRRTL, etc.
  }];
  let constructor = "ksim::createLoadFIRRTLPass()";
}

def CleanDesign : Pass<"ksim-clean-design", "circt::ModuleOp"> {
  let summary = "Clean chipyard design";
  let description = [{
    This pass will remove plusarg reader, replace EICG_wrapper into compreg.ce
  }];
  let constructor = "ksim::createCleanDesignPass()";
}

def TemporalFusion : Pass<"ksim-temporal-fusion", "circt::hw::HWModuleOp"> {
  let summary = "Temporal Fusion Algorithm for KSim";
  let description = summary;
  let constructor = "ksim::createTemporalFusionPass()";
  let options = [
    Option<"tol", "tol", "int", "10", "Maximum iteration to solve the programming problem">,
    Option<"verbose", "verbose", "bool", "false", "Verbose optimization log">,
    Option<"graphOut", "graph-out", "std::string", "", "output graph dot visualization">,
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">
  ];
}

def LowerState : Pass<"ksim-lower-state", "circt::ModuleOp"> {
  let summary = "Lower state component to ksim.low.queue op";
  let description = summary;
  let constructor = "ksim::createLowerStatePass()";
  let options = [
    Option<"headerFile", "header", "std::string", "", "output header file">,
    Option<"driverFile", "driver", "std::string", "", "output driver file">,
    Option<"emitComb", "emitComb", "bool", "false", "emit combinational logic">,
    Option<"prefix", "prefix", "std::string", "", "variable prefix">,
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">
  ];
}

def LowerToLLVM : Pass<"ksim-lower-to-llvm", "circt::ModuleOp"> {
  let summary = "Lower sequential logic to llvm dialect";
  let description = summary;
  let constructor = "ksim::createLowerToLLVMPass()";
  let options = [
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">
  ];
}

def CombToLLVM : Pass<"ksim-comb-to-llvm", "circt::ModuleOp"> {
  let summary = "Lower combinational logic to llvm dialect without changing module";
  let description = summary;
  let constructor = "ksim::createCombToLLVMPass()";
}

def AddLLVMDebugInfo : Pass<"ksim-add-debug-info", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Add LLVM debug info";
  let description = summary;
  let constructor = "ksim::createAddLLVMDebugInfoPass()";
  let options = [];
}

#endif
