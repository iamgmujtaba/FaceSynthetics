## Introduction
This repository provides the 3D 68 Keypoints from the synthetic dataset from Microsoft [Fake It Till You Make It: Face analysis in the wild using synthetic data alone](https://arxiv.org/abs/2109.15102).

## Installation
```
conda create -n synthetic python=3.6 -y
conda activate synthetic 
```

To install PyTorch along with CUDA run the following script
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

To install packages remaining packages, run the following script.
```
pip install -r .\requirements.txt
```

If you are getting the following error 
```
app = FaceAnalysis()
TypeError: __init__() missing 1 required positional argument: 'name'
```
run the following script
```
pip install -U insightface
```

## Dataset
Download `Face Synthetics dataset` from [https://github.com/microsoft/FaceSynthetics](https://github.com/microsoft/FaceSynthetics) and put it somewhere.

<div align="left">
  <img src="https://github.com/microsoft/FaceSynthetics/raw/main/docs/img/dataset_samples_2.jpg" width="640"/>
</div>
<br/>

Then use [prepare_dataset.py](prepare_dataset.py) for training data preparation.


## Train and Test

### Training
`` python train.py `` which uses `resnet50d` as backbone by default, please check the [code](train.py) for detail.

#### Pretrained Model
I have trained the network for 1000 epochs. The pre-trained model can be obtained from google drive
[ResNet50d](https://drive.google.com/file/d/1lglDZypW_1ihWsfCOGS_mNCyaQryAvlb/view?usp=sharing). Place the pretrained model on [models][models] directory. Run [test.py](test.py) for the restuls. 


### Testing
Please check [test.py](test.py) for detail.

## Result Visualization(3D 68 Keypoints)
<div align="left">
  <img src="https://github.com/iamgmujtaba/FaceSynthetics/blob/main/doc/a_indoor_001.png?raw=true" width="320"/>
  <img src="https://github.com/iamgmujtaba/FaceSynthetics/blob/main/doc/a_indoor_009.png?raw=true" width="320"/>
</div>

<div align="left">
  
</div>

<div align="left">
  <img src="https://github.com/iamgmujtaba/FaceSynthetics/blob/main/doc/a_outdoor_008.png?raw=true" width="320"/>
</div>

<div align="left">
  <img src="https://github.com/iamgmujtaba/FaceSynthetics/blob/main/doc/a_outdoor_010.png?raw=true" width="320"/>
</div>

## Acknowledgment
The base code is borrowed from [insightface](https://github.com/deepinsight/insightface).
