ARG py_suffix="py37"
FROM yellow.hub.cambricon.com/mmcv/base/x86_64/mmcv:v0.1-x86_64-manylinux2014-mmcv_${py_suffix}_gcc7_mpi

ARG mmcv_branch="dev-2.1"
ARG catch_version="1.16.0"
ARG torch_version="1.9"
ARG torch_full_version="1.9.0"
ARG python_version="3.7"

RUN mkdir /torch /wheel

# fetch catch codes
RUN pushd /torch && \
    wget http://daily.software.cambricon.com/release/pytorch${torch_version}/${catch_version}/src/pytorch-v${catch_version}-torch${torch_version}.tar.gz && \
    tar -xvf pytorch-v${catch_version}-torch${torch_version}.tar.gz && \
    popd

# fetch mmcv codes
RUN pushd /torch && \
    echo $mmcv_branch && \
    git clone http://gitlab.software.cambricon.com/neuware/oss/openmmlab/mmcv.git -b $mmcv_branch && \
    popd

RUN pushd /torch && bash /torch/cambricon_pytorch/pytorch/src/catch/script/release/independent_build.sh -r dep -o centos -v 7

ENV NEUWARE_HOME=/torch/neuware_home
ENV LD_LIBRARY_PATH=/torch/neuware_home/lib64:$LD_LIBRARY_PATH
ENV CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
ENV CXX=/opt/rh/devtoolset-7/root/usr/bin/g++
ENV TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -fvisibility=hidden -Wl,--exclude-libs,ALL"
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV GLIBCXX_USE_CXX11_ABI=0
ENV MMCV_WITH_OPS=1

# set pip source
RUN PIP_CONF_DIR=${HOME}/.config/pip && mkdir -p ${PIP_CONF_DIR} && \
    echo "[global]" > ${PIP_CONF_DIR}/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple/" \
          >> ${PIP_CONF_DIR}/pip.conf && \
    echo "extra-index-url = http://mirrors.cambricon.com/pypi/web/simple/" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "[install]" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ${PIP_CONF_DIR}/pip.conf && \
    echo "               mirrors.cambricon.com" >> ${PIP_CONF_DIR}/pip.conf

# build mmcv
ENV CPLUS_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$CPLUS_INCLUDE_PATH
ENV C_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$C_INCLUDE_PATH
ENV py_version="37"

# set env for bangc compile
ENV TORCH_BANG_ARCH_LIST="3.0;5.0"

RUN pushd /torch && \
    /opt/py${python_version}/bin/pip${python_version} install virtualenv && \
    /opt/py${python_version}/bin/virtualenv -p /opt/py${python_version}/bin/python${python_version} venv3/pytorch && \
    source venv3/pytorch/bin/activate && \
    wget http://daily.software.cambricon.com/release/pytorch${torch_version}/${catch_version}/wheel/torch-${torch_full_version}-cp${py_version}-cp${py_version}m-linux_x86_64.whl && \
    pip install torch-${torch_full_version}-cp${py_version}-cp${py_version}m-linux_x86_64.whl && \
    wget http://daily.software.cambricon.com/release/pytorch${torch_version}/${catch_version}/wheel/torch_mlu-${catch_version}_torch${torch_version}-cp${py_version}-cp${py_version}m-linux_x86_64.whl && \
    pip install torch_mlu-${catch_version}_torch${torch_version}-cp${py_version}-cp${py_version}m-linux_x86_64.whl && \
    pushd /torch/mmcv && \
    pip install -r requirements/torch.txt && \ 
    pip install -r requirements.txt && \
    python setup.py bdist_wheel && \
    cp /torch/mmcv/dist/*.whl /wheel/
