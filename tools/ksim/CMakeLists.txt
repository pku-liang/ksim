add_llvm_tool(ksim ksim.cpp)
llvm_update_compile_flags(ksim)
target_link_libraries(ksim PRIVATE 
  KSim
  KSimTransforms

  CIRCTHW
  CIRCTComb
  CIRCTSeq

  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRTransforms
  MLIRSupport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  MLIRTargetLLVMIRExport
  MLIRLLVMIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
