# OCT_Classification
Classifying different Retinal Diseases using Deep Learning from Optical Coherence Tomography Images

# Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card

# Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers] (https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)
- Install Pip and Python3 enviornment
```
sudo apt-get install pip3 python3-dev
```
- Install  packages from requirements.txt
```
sudo pip3 -r requirements.txt
```
# Installation Instruction for Windows
- Download and Install [Nvidia Drivers](https://www.nvidia.com/download/driverResults.aspx/130631/en-us)
- Download and Install [Nvidia Cuda Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=7&target_type=exelocal)
- Install Pip and Python3 environemnt from [Web link](https://www.python.org/downloads/windows/)
- Install packages from requirements.txt
```
sudo pip3 -r requirements.txt
```

# Demo
```
python3 inference.py --imgpath='location of the testing image(single file)' --weights='location to the .h5 file' --dataset='Srinivasan2014 or Kermany2018 (mention either of the dataset'
```

# Training on OCT2017 Dataset

Please cite the paper if you use their data
```
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and McKeown, Alex and Yang, Ge and Wu, Xiaokang and Yan, Fangbing and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
```

- Dataset download link 
```
https://data.mendeley.com/datasets/rscbjbr9sj/3
```

- Folder structure for training
```
├── data
|   ├──OCT2017
|       ├──train
|           ├──CNV
|           ├──DME
|           ├──DRUSEN
|           └──Normal
|       ├──test
|           ├──CNV
|           ├──DME
|           ├──DRUSEN
|           └──Normal
├── src
├── LICENSE
├── README.md
├── inference.py
├── requirements.txt
├── test.py
└── train.py
```
- Type this in terminal to run the train.py file
```
python3 train.py --dataset=Kermany2018 --datadir=OCT2017 --batch=4 --epoch=30 --logdir=optic-net-log --snapshot_name=optic-net-custom
```
- There are different flags to choose from. Not all of them are mandatory

```
   '--dataset', type=str, required=True, help='Choosing between 2 OCT datasets', choices=['Srinivasan2014','Kermany2018']
   '--batch', type=int, default=8
   '--input_dim', type=int, default=224
   '--datadir', type=str, required=True, help='path/to/data_directory'
   '--epoch', type=int, default=30
   '--logdir', type=str
   '--weights', type=str,default=None, help='Resuming training from previous weights'
   '--snapshot_name',type=str, default=None, help='Name the saved snapshot'
```
