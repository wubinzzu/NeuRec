#!/bin/bash
set -e -u -x

# Compile cpp
cd io/

dir_name=${SO_DIR}
mkdir ${dir_name}

#mkdir compiled_so
for PYBIN in /opt/python/*/bin; do
    if [[ ${PYBIN} =~ "3" ]]
    then
        echo ${PYBIN}
        PYTHON_EXE="${PYBIN}/python"
        PIP_EXE="${PYBIN}/pip"
        ${PIP_EXE} install cython
        ${PIP_EXE} install numpy
        ${PYTHON_EXE} setup.py build_ext --inplace clean --all

        find . -path ./${dir_name} -prune -o -name '*.so' -print |xargs tar czf ${dir_name}.tgz
        tar zxvf ${dir_name}.tgz -C ${dir_name}
    fi

done
