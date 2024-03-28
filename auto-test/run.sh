echo threads,run,exe,time
for f in ./runs/*/; do
    threads=${f##./runs/ksim-}
    threads=${threads%%/}
    for exe in $f/bin/*.exe; do
        exe_name=${exe##*/}
        exe_name=${exe_name%%.exe}
        for ((i=0;i<5;i++)); do
            echo -n $threads,$i,$exe_name,
            taskset -c $(seq -s, 1 $threads) $f/bin/SmallBoomCore.exe 1000
        done
    done
done
