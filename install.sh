#!/bin/bash

conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -y tqdm
pip install PyYAML
pip install easydict
pip install cython
pip install pycocotools
pip install opencv-python
pip install tb-nightly
pip install pandas
pip install jpeg4py
pip install tikzplotlib
pip install colorama
pip install scipy
pip install visdom
pip install tensorboardX
pip install timm
pip install wandb
pip install setuptools
