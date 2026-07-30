#!/bin/bash

[[ -f Makefile ]] && make clean

rm -rf CMakeCache.txt  CMakeFiles  cmake_install.cmake  CTestTestfile.cmake  Makefile  yakl  \
       core  modules  custom_modules  inputs  model  libcustom_modules.a  cmake_packages     \
       generated  git-state.txt  Testing

