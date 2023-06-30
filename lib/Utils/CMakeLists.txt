add_circt_dialect_library(
  KSimLLVMQueue
  LLVMQueue.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  CIRCTHW
)