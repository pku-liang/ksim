#!/bin/bash
start_cpu=16
rg8() {
    seq $1 $(($1+7))
}
threads=($(rg8 1) $(rg8 129) $(rg8 9) $(rg8 137))
nthread() {
    res=$(printf %s, "${threads[@]:0:$1}")
    echo ${res%%,}
}
echo threads,run,exe,time
for f in ./runs/*/; do
    th=${f##./runs/ksim-}
    th=${th%%/}
    for exe in $f/bin/*.exe; do
        exe_name=${exe##*/}
        exe_name=${exe_name%%.exe}
        for ((i=0;i<50;i++)); do
            echo -n $th,$i,$exe_name,
            taskset -c $(nthread $th) $exe 10000
        done
    done
done
