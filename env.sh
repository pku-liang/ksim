export PATH=$PWD/install/bin:$PATH
export KSIM_ROOT=$PWD

build() {
    make -C $KSIM_ROOT/build -j4
}