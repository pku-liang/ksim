add_circt_dialect_library(
  KSim

  KSimDialect.cpp
  KSimTypes.cpp

  DEPENDS
  MLIRKSimIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRSupport
)

add_subdirectory(Transforms)
add_subdirectory(Utils)