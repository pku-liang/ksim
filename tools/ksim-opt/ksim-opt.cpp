#include "ksim/KSimDialect.h"
#include "ksim/KSimPasses.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


using namespace mlir;
using namespace circt;

int main(int argc, char ** argv) {
  mlir::DialectRegistry registry;
  registry.insert<hw::HWDialect>();
  registry.insert<seq::SeqDialect>();
  registry.insert<sv::SVDialect>();
  registry.insert<ksim::KSimDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<func::FuncDialect>();
  ksim::registerPasses();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerControlFlowSinkPass();
  LLVM::registerDIScopeForLLVMFuncOpPass();
  return mlir::failed(mlir::MlirOptMain(argc, argv, "ksim driver", registry));
}
