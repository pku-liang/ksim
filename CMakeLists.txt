cmake_minimum_required(VERSION 3.13.4)

project(ksim VERSION 0.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(AddCIRCT)

add_compile_options(-fexceptions -Wno-suggest-override -Wno-register)
add_compile_options(-g)
add_compile_options(-O2)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CIRCT_MAIN_INCLUDE_DIR} ${CIRCT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(third_party)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

install(TARGETS ksim DESTINATION bin)
install(TARGETS ksim-opt DESTINATION bin)