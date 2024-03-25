#!/bin/bash

chmod -R 777 mlu-ops

pushd mlu-ops
source env.sh
./independent_build.sh --no_prepare --disable-gtest --enable-static
popd

if [ -d "./mmcv/lib" ]; then
    echo "mmcv/lib directory already existed!"
else
    echo "Creating mmcv/lib directory!"
    mkdir mmcv/lib
fi
cp mlu-ops/build/lib/*.a* ./mmcv/lib
