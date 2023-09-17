# Khronos

Khronos is moving to [https://github.com/pku-liang](https://github.com/pku-liang).

## Introductions

## Installation

Setup depedencies:

```bash
mkdir install
export INSTALL_PREFIX=$PWD/install
git submodule update --init
cd third_party
./setup-circt.sh
./setup-lemon.sh
```

Build khronos:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release
make ksim
make install
```

## Usage



## Citing Khronos

If you use this software, please cite it as below.

```bibtex
@inproceedings{Khronos,
  author = {Kexing, Zhou and Yun, Liang and Yibo, Lin and Runsheng, Wang and Ru, Huang},
  title = {Khronos: Fusing Memory Access for Improved Hardware RTL Simulation},
  booktitle = {MICRO '23: 55th IEEE/ACM International Symposium on Microarchitecture},
  publisher = {ACM},
  year = {2023},
  url = {https://doi.org/10.1145/3613424.3614301},
  doi = {10.1145/3613424.3614301}
}
```
