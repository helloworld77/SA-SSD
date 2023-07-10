### 切gcc 和 cuda 版本
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
gcc-5
export CUDA_HOME=/usr/local/cuda-10.1
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
## 创建 python
conda create -n sassd_orign python=3.7
conda activate sassd_orign
## torch版本不能太高，官方推荐1.1.0
pip install ~/software/torch/torch-1.3.0+cu100-cp37-cp37m-linux_x86_64.whl ~/software/torch/torchvision-0.4.1+cu100-cp37-cp37m-linux_x86_64.whl
## 
pip install pybind11 Cython
### numba 死活不过，还是删掉 numba jit 吧。
pip install numba==0.55.1
pip install numpy==1.21.5
pip install Pillow==4.2.0 matplotlib==3.0 pycocotools imageio==2.9.0 terminaltables
pip install mmcv-full==1.1.0 ##走源码安装更安全 #-f https://download.openmmlab.com/mmcv/dist/cu100/torch1.3.0/index.html

## 编译
find . -name build|xargs rm -rf
find . -name *.so|xargs rm -rf
cd mmdet/ops/points_op/
python setup.py build_ext --inplace
cd ../iou3d/
python setup.py build_ext --inplace
cd ../pointnet2/
python setup.py build_ext --inplace

## spconv 1.x: git clone https://github.com/traveller59/spconv.git --recursive
rm -rf build
python setup.py bdist_wheel
pip install dist/spconv-1.1-cp37-cp37m-linux_x86_64.whl
###############spconv 1.x: cmake find torch error:
##### vim /opt/miniconda3/envs/sassd/lib/python3.7/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake\
##### comment line 158:
#   if(CUDNN_VERSION VERSION_LESS "7.0.0")
#     message(FATAL_ERROR "PyTorch requires cuDNN 7 and above.")
#   endif()
############### cmake find torch error:

## create dataset
export PYTHONPATH=/home/ssd1t/workspace/projects/3d/SA-SSD
python tools/create_data.py

## training
cd tools
export PYTHONPATH=`pwd`
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
python tools/train.py configs/car_cfg.py
python tools/train.py configs/multi_cfg.py
### multi gpu
bash dist_train.sh
