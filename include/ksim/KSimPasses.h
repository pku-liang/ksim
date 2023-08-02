#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace ksim {

#define GEN_PASS_DECL
#include "ksim/Passes.h.inc"

std::unique_ptr<mlir::Pass> createIsolateTopModulePass(IsolateTopModuleOptions options=IsolateTopModuleOptions());
std::unique_ptr<mlir::Pass> createAddZeroResetPass();
std::unique_ptr<mlir::Pass> createCombToLLVMPass();
std::unique_ptr<mlir::Pass> createSlicePropagationPass();
std::unique_ptr<mlir::Pass> createFlattenPass();
std::unique_ptr<mlir::Pass> createDCEPass();
std::unique_ptr<mlir::Pass> createMiliToMoorePass();
std::unique_ptr<mlir::Pass> createLoadFIRRTLPass();
std::unique_ptr<mlir::Pass> createCleanDesignPass();
std::unique_ptr<mlir::Pass> createTemporalFusionPass(TemporalFusionOptions options=TemporalFusionOptions());
std::unique_ptr<mlir::Pass> createTemporalFusion2Pass(TemporalFusion2Options options=TemporalFusion2Options());
std::unique_ptr<mlir::Pass> createRemoveSVPass();
std::unique_ptr<mlir::Pass> createLowerStatePass(LowerStateOptions options=LowerStateOptions());
std::unique_ptr<mlir::Pass> createLowerToLLVMPass(LowerToLLVMOptions options=LowerToLLVMOptions());
std::unique_ptr<mlir::Pass> createAddLLVMDebugInfoPass();
std::unique_ptr<mlir::Pass> createDumpCombPass(DumpCombOptions options=DumpCombOptions());

#define GEN_PASS_REGISTRATION
#include "ksim/Passes.h.inc"

}