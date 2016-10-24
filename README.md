[![Build Status](https://travis-ci.org/choderalab/sams.svg?branch=master)](https://travis-ci.org/choderalab/sams)

# Self-adjusted mixture sampling (SAMS)

Self-adjusted mixture sampling experiments, based on ideas from [Zhiqiang Tan (Rutgers)](http://stat.rutgers.edu/~ztan/)

## Notice

Please be aware that this code is made available in the spirit of open science, but is currently pre-alpha--that is, it is not guaranteed to be completely tested or provide the correct results, and the API can change at any time without warning. If you do use this code, do so at your own risk. We appreciate your input, including raising issues about potential problems with the code, but may not be able to address your issue until other development activities have concluded.

## References
* [1] Tan, Z. (2015) [Optimally adjusted mixture sampling and locally weighted histogram analysis](http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf), Journal of Computational and Graphical Statistics, to appear.

## Authors
* [John D. Chodera](http://choderalab.org) (MSKCC)
* [Zhiqiang Tan](http://stat.rutgers.edu/home/ztan/) (Rutgers)

## Getting started

Make sure you have the `omnia` channel and `dev` label added to your [`conda`](http://conda.pydata.org/) path:
```bash
# Add omnia channel for stable omnia packages
conda config --add channels omnia
# Add omnia dev channels to pick up `openmm 7.1.0` pre-release
conda config --add channels omnia/label/dev
```
Install `sams`:
```bash
conda install --yes sams
```
Try the example in `examples/abl-imatinib-explicit/`:
```bash
cd examples/abl-imatinib-explicit/
python soften-ligand.py
```
