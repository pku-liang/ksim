#!/bin/bash

n=$1

create() {
    i=$1
    rm -rf ksim-$i
    cp -r --reflink=auto ksim-1 ksim-$i
    sed -e "s#threads=.*#threads=$i#g" -i ksim-$i/compile.sh
}

for i; do
    if [ $i -gt 1 ]; then
        echo create $i
        create $i
    fi
done

