add_llvm_tool(ksim-opt ksim-opt.cpp)
llvm_update_compile_flags(ksim-opt)
target_link_libraries(ksim-opt PRIVATE 
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
  MLIROptLib
  MLIRLLVMIRTransforms
)
