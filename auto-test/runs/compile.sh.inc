#!/bin/bash

compile() {
    threads=$1
    file=$2
    base=${file%%.mlir}
    env FILENAME=$base envsubst < $KSIM_ROOT/rt/rt.cpp > $base-final.cpp
    [ -f $base.ll ] || $KSIM_ROOT/build/bin/ksim $file --parallel $threads --out-driver=$base.cpp -o $base.ll 2>compile.log
    [ -f $base.o ]  || llc -filetype=obj -mtriple=x86_64-pc-linux-gnu -relocation-model=pic $base.ll -o $base.o
    g++ -fopenmp -O2 $base.cpp $base.o -o $base -lpthread
}
