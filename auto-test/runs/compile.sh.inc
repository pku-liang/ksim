#!/bin/bash

compile() {
    threads=$1
    file=$2
    base=${file%%.mlir}
    env FILENAME=$base envsubst < $KSIM_ROOT/rt/rt.cpp > $base-final.cpp
    [ -f $base.ll ] || $KSIM_ROOT/build/bin/ksim $file --parallel $threads --out-header=$base.h --out-driver=$base.cpp --out-par-header=$base.par.h -o $base.ll
    [ -f $base.o ]  || clang++ -O1 $base.ll -c -o $base.o
    clang++ -O1 -std=c++20 $base-final.cpp $base.o -o $base -lpthread
}
