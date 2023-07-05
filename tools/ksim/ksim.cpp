#include "ksim/KSimDialect.h"
#include "ksim/KSimPasses.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

enum Level {
  CoreIR = 0,
  Flattened = 1,
  KSimFused = 2,
  KSimLow = 3,
  LLVMDialect = 4,
  LLVMIR = 5
};

static cl::OptionCategory mainCategory("ksim Options");
const auto LevelClasses = cl::values(
    clEnumValN(CoreIR, "core", "Default circt core IR"),
    clEnumValN(Flattened, "flattened", "Ksim high level ir"),
    clEnumValN(KSimFused, "fused", "Ksim fused ir"),
    clEnumValN(KSimLow, "low", "KSim low level ir"),
    clEnumValN(LLVMDialect, "llvm", "LLVM dialect"),
    clEnumValN(LLVMIR, "llvmir", "LLVM IR")
  );
static cl::opt<Level> inputLevel(
  "in", cl::desc("Input file type"), 
  LevelClasses, cl::init(CoreIR), cl::cat(mainCategory));
static cl::opt<Level> outputLevel(
  "out", cl::desc("Output file type"),
  LevelClasses, cl::init(LLVMIR), cl::cat(mainCategory));
static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"), cl::cat(mainCategory));
static cl::opt<std::string> outputFilename("o", cl::desc("Output file name"), cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));
static cl::opt<bool> emitBytecode("emit-bytecode",
  cl::desc("Emit bytecode when generating MLIR output"),
  cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> disableOptimizations("disable-optimizations", cl::desc("disable optimizations"), cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> outputHeaderFilename("out-header", cl::desc("output header file name"), cl::value_desc("filename"), cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> outputDriverFilename("out-driver", cl::desc("output driver file name"), cl::value_desc("filename"), cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> outputGraphFilename("out-graph", cl::desc("output graph file name"), cl::value_desc("filename"), cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> emitVarPrefix("prefix", cl::desc("emit variable prefix"), cl::value_desc("name"), cl::init(""), cl::cat(mainCategory));
static cl::opt<bool> emitComb("emit-comb", cl::desc("emit combinational eval func"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> emitDriver("a", cl::desc("emit driver, will automatically fill output, out-header and out-driver"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> verbose("v", cl::desc("verbose"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> generateDebugInfo("g", cl::desc("debug"), cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> forceZeroReset("force-zero-reset", cl::desc("force register with no reset value with zero reset"), cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> dumpGraph("dump-graph", cl::desc("dump graph"), cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> CXXFLAGS("CXXFLAGS", cl::desc("Generated CXXFLAGS"), cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> LIBS("LIBS", cl::desc("Generated LIBS"), cl::init(""), cl::cat(mainCategory));
static std::string programName;
static cl::opt<bool> disableClockGate("disable-clock-gate", cl::desc("Disable clock gating optimization"), cl::init(false), cl::cat(mainCategory));

static void printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode) {
    writeBytecodeToFile(op, os, mlir::BytecodeWriterConfig("ksim"));
  }
  else {
    OpPrintingFlags flag;
    if(generateDebugInfo)
      flag.enableDebugInfo(true, false);
    flag.useLocalScope();
    op->print(os, flag);
  }
}
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::unique_ptr<llvm::ToolOutputFile> &output) {
  auto mod = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if(!mod) return failure();
  PassManager pm(&context);
  if(failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(ts);
  if(inputLevel < Flattened && Flattened <= outputLevel) {
    pm.addPass(ksim::createRemoveSVPass());
    pm.addPass(ksim::createFlattenPass());
    pm.addPass(ksim::createCleanDesignPass());
    if(!dumpGraph.empty()) {
      ksim::DumpCombOptions options;
      options.file = dumpGraph;
      pm.addPass(ksim::createDumpCombPass(options));
    }
    pm.addPass(ksim::createLoadFIRRTLPass());
    if(forceZeroReset) {
      pm.addNestedPass<hw::HWModuleOp>(ksim::createAddZeroResetPass());
    }
    pm.addPass(mlir::createSymbolDCEPass());
    if(!disableOptimizations) {
      pm.addNestedPass<hw::HWModuleOp>(ksim::createDCEPass());
    }
  }
  if(inputLevel < KSimFused && KSimFused <= outputLevel) {
    ksim::TemporalFusionOptions options;
    options.disableOptimization = disableOptimizations;
    options.verbose = verbose;
    options.graphOut = outputGraphFilename;
    options.disableClockGate = disableClockGate;
    pm.addNestedPass<hw::HWModuleOp>(ksim::createTemporalFusionPass(options));
  }
  if(inputLevel < KSimLow && KSimLow <= outputLevel) {
    ksim::LowerStateOptions options;
    options.headerFile = outputHeaderFilename;
    options.driverFile = outputDriverFilename;
    options.emitComb = emitComb;
    options.prefix = emitVarPrefix;
    options.disableOptimization = disableOptimizations;
    pm.addPass(ksim::createLowerStatePass(options));
    if(!disableOptimizations) {
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createCanonicalizerPass());
    }
  }
  if(inputLevel < LLVMDialect && LLVMDialect <= outputLevel) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    if(generateDebugInfo) {
      pm.addNestedPass<LLVM::LLVMFuncOp>(LLVM::createDIScopeForLLVMFuncOpPass());
      pm.addNestedPass<LLVM::LLVMFuncOp>(ksim::createAddLLVMDebugInfoPass());
    }
    ksim::LowerToLLVMOptions options;
    options.disableOptimization = disableOptimizations;
    pm.addPass(ksim::createLowerToLLVMPass(options));
    pm.addPass(ksim::createCombToLLVMPass());
  }
  if(failed(pm.run(mod.get())))
    return failure();
  if(outputLevel <= LLVMDialect) {
    auto outputTimer = ts.nest("Print .mlir output");
    printOp(*mod, output->os());
    output->keep();
    return success();
  }
  else {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(mod.get(), llvmContext);
    if (!llvmModule) return failure();
    auto outputTimer = ts.nest("Print .ll output");
    llvmModule->print(output->os(), nullptr, false, true);
    output->keep();
    return success();
  }
}

static std::string getReplacedFilename(std::string name, std::string ext) {
  SmallString<128> out(name);
  sys::path::replace_extension(out, ext);
  return out.str().str();
}

static void emitMakefile(std::string name) {
  std::error_code ec;
  raw_fd_ostream fout(name, ec);
  if(ec) return;
  auto objFilename = getReplacedFilename(name, ".o");
  auto tbFilename = getReplacedFilename(name, "tb");
  fout << "KSIM?=" << programName << "\n";
  fout << "KSIMFLAGS+=";
  if(verbose) fout << " -v";
  if(generateDebugInfo) fout << " -g";
  if(emitComb) fout << " --emit-comb";
  if(!outputGraphFilename.empty()) fout << " --out-graph " << outputGraphFilename;
  if(!outputDriverFilename.empty()) fout << " --out-driver " << outputDriverFilename;
  if(!outputHeaderFilename.empty()) fout << " --out-header " << outputHeaderFilename;
  if(disableOptimizations) fout << " --disable-optimizations";
  if(!emitVarPrefix.empty()) fout << " --prefix " << emitVarPrefix;
  fout << "\n";
  fout << "CXXFLAGS+=" << CXXFLAGS << "\n";
  fout << "LIBS+=" << LIBS << "\n";
  fout << tbFilename << " : " << objFilename << " " << outputDriverFilename << "\n";
  fout << "\t" << "clang++ $(CXXFLAGS) " << objFilename << " " << outputDriverFilename << " $(LIBS) -o $@\n";
  fout << "\n";
  fout << outputFilename << " &: " << inputFilename << "\n";
  fout << "\t" << "$(KSIM) -a $(KSIMFLAGS) $^\n";
  fout << "\n";
  fout << objFilename << " : " << outputFilename << "\n";
  fout << "\t" << "llc --filetype=obj $(LLCFLAGS) $^ -o $@\n";
  fout << "\n";
  fout << ".phony: clean\n";
  fout << "clean: \n";
  fout << "\trm -rf " << objFilename << " " << tbFilename << " " << outputHeaderFilename << outputFilename << "\n";
}

static LogicalResult executeKSim(MLIRContext &context) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  if(emitDriver) {
    if(outputFilename != "-") {
      llvm::errs() << "output file name is ignored" << "\n";
    }
    outputFilename = getReplacedFilename(inputFilename, "ll");
    if(outputHeaderFilename.empty())
      outputHeaderFilename = getReplacedFilename(inputFilename, "h");
    if(outputDriverFilename.empty())
      outputDriverFilename = getReplacedFilename(inputFilename, "cpp");
    auto outputMakefile = getReplacedFilename(inputFilename, "mk");
    if(!sys::fs::exists(outputMakefile)) {
      emitMakefile(outputMakefile);
    }
    if(sys::fs::exists(outputDriverFilename)) {
      outputDriverFilename = "";
    }
  }

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);

  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  context.loadDialect<
    hw::HWDialect,
    seq::SeqDialect,
    sv::SVDialect,
    ksim::KSimDialect,
    comb::CombDialect,
    LLVM::LLVMDialect,
    func::FuncDialect,
    mlir::DLTIDialect
  >();
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream & os) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
    context.printOpOnDiagnostic(false);
    (void)processBuffer(context, ts, sourceMgr, output);
    return sourceMgrHandler.verify();
  };
  return chunkFn(std::move(input), llvm::outs());
  // return splitAndProcessBuffer(std::move(input), chunkFn, llvm::outs());
}

int main(int argc, char ** argv) {
  programName = argv[0];
  llvm::InitLLVM y(argc, argv);
  MLIRContext context;
  cl::HideUnrelatedOptions(mainCategory);
  ksim::registerPasses();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerControlFlowSinkPass();
  LLVM::registerDIScopeForLLVMFuncOpPass();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "KSim Compiler");
  auto result = executeKSim(context);
  exit(failed(result));
}
