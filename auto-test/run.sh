#!/bin/bash
start_cpu=16
threads=($(seq 1 32))
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
        for ((i=0;i<10;i++)); do
            echo -n $th,$i,$exe_name,
            taskset -c $(nthread $th) $exe 10000
        done
    done
done
