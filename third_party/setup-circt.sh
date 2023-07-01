#!/bin/bash

INSTALL_PREFIX=$(realpath ../install)
[ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX

cd circt
git submodule update --init

pushd llvm

[ -d build ] || mkdir build
cd build
cmake ../llvm \
  -GNinja \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \

ninja install

popd

[ -d build ] || mkdir build
cd build
cmake .. \
  -GNinja \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DESI_COSIM=OFF -DESI_CAPN=OFF
ninja install