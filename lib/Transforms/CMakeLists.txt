set(LLVM_TARGET_DEFINITIONS SlicePropagation.td)
mlir_tablegen(SlicePropagation.cpp.inc -gen-rewriters)
add_public_tablegen_target(KSimSlicePropagationIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_circt_dialect_library(
  KSimTransforms

  CleanDesign.cpp
  IsolateTopModule.cpp
  AddZeroReset.cpp
  AddDebugInfo.cpp
  CombToLLVM.cpp
  Flatten.cpp
  LoadFIRRTL.cpp
  TemporalFusion.cpp
  RemoveSV.cpp
  LowerToLLVM.cpp
  LowerState.cpp
  SlicePropagation.cpp
  DCE.cpp
  MiliToMoore.cpp
  DumpComb.cpp

  DEPENDS
  KSimTransformsIncGen
  KSimSlicePropagationIncGen

  LINK_LIBS PUBLIC

  KSim
  KSimLLVMQueue

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTHWToLLVM
  CIRCTCombToLLVM

  MLIRPass
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts

  lemon
)
