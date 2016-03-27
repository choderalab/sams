# Continuous integration boilerplate

## Manifest
* `travis-ci/` - continuous integration with travis
* `appveyor/` - continuous integration with appveyor

## Example `.travis.yml`
```YAML
language: c
sudo: false
addons:
  apt:
    sources:
    - ubuntu-toolchain-r-test

install:
  - bash -x devtools/travis-ci/install.sh
  - export PYTHONUNBUFFERED=true
  - export PATH=$HOME/miniconda/bin:$PATH

script:
  # Add conda channel
  - conda config --add channels $ORGNAME
  # Build the recipe
  - conda build devtools/conda-recipe
  # Test the local installation
  - source activate _test
  - conda install --yes --quiet nose nose-timer
  - cd devtools && nosetests $PACKAGENAME --nocapture --verbosity=2 --with-doctest --with-timer -a "\!slow" && cd ..

env:
  matrix:
    - python=2.7  CONDA_PY=27
    - python=3.4  CONDA_PY=34
    - python=3.5  CONDA_PY=35

  global:
    - ORGNAME="omnia" # the name of the organization
    - PACKAGENAME="packagename" # the name of your package
    # encrypted BINSTAR_TOKEN for push of dev package to binstar
    # - secure: "RRvLDPu9mPoNaRWIseaJdgShOXI+PaHPWKAIJvW7VYWcAS6iEN7W4Fj4zD5hkocQxc3ou97EtkgID+ApH10bSGKxCykyU0urSY9jsSOJX2m0AE19X0dVr6ySIQkwCWE6kUMVlvQYQo80fM2EMElD+btr4G9XBAhArAO7HvZHkoQ="

after_success:
  - echo "after_success"
  - if [ "$TRAVIS_SECURE_ENV_VARS" == true ]; then source devtools/travis-ci/after_success.sh; fi
```
