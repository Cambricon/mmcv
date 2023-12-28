编译安装
===============

二进制wheel包安装
++++++++++++++++++++++++
1. 参考寒武纪PyTorch⽤⼾⼿册准备PyTorch环境。

2. 获取Cambricon MMCV二进制wheel安装包。

3. 安装二进制wheel安装包。

   ::

     pip install mmcv-2.1.0+v{xxx}-cp{xxx}-cp{xxx}-linux_x86_64.whl

源码编译安装
++++++++++++++++++++++++
1. 参考寒武纪PyTorch⽤⼾⼿册准备PyTorch环境。

2. 获取Cambricon MMCV源码包并解压。

3. 安装requirements目录下的编译相关的依赖包。

   ::

     pip install -r requirements/build.txt #安装第三方包

4. 编译安装Cambricon MMCV。

   ::

     python setup.py install

软件环境依赖
+++++++++++++++++

.. list-table:: 支持的版本组合列表
    :widths: 40 40
    :header-rows: 1

    * - Pytorch版本
      - Python版本

    * - 1.9
      - 3.7

    * - 1.13
      - 3.10

