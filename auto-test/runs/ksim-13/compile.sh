#!/bin/bash

threads=13

. ../compile.sh.inc

cd $1; shift

file=$1
outfile=$2
base=${file%%.mlir}

firtool --ir-hw --disable-all-randomization $file -o $base.mlir
compile $threads $base.mlir
cp $base $outfile
