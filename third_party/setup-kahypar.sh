#!/bin/bash

if [[ "$INSTALL_PREFIX" == "" ]]; then
  INSTALL_PREFIX=$(realpath ../install)
  [ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX
fi

cd kahypar
[ -f CMakeLists.txt.bak ] || sed -i.bak -e '/add_subdirectory(tools)/d' CMakeLists.txt
[ -f .gitignore.bak ] || sed -i.bak -e '1ibuild' .gitignore


mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cp kahypar/application/KaHyPar $INSTALL_PREFIX/bin/KaHyPar