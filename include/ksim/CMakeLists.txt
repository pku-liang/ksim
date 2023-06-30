add_circt_dialect(KSim ksim)
add_circt_dialect_doc(Seq seq)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(KSimTransformsIncGen)