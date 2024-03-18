export PATH=$PWD/install/bin:$PATH
export KSIM_ROOT=$PWD

build() {
    make -C $KSIM_ROOT/build -j$(nproc) ksim ksim-opt
}

run-ksim() {
    base=${1%%.mlir}
    env FILENAME=$base envsubst < $KSIM_ROOT/rt/rt.cpp > $base-final.cpp
    $KSIM_ROOT/build/bin/ksim $1 --parallel 4 --out-header=$base.h --out-driver=$base.cpp --out-par-header=$base.par.h -o $base.ll
}
