# HYU 2022 AI system design Final project

## Junhgo Kim, Junyong Yun
### LSQ on Monocular 3D Object Detection

- M3D-RPN: Monocular 3D Region Proposal Network for Object Detection, Garrick, 2019, CVPR
- Learned step size quantization, Esser, 2019, arXiv

## Introduction

Our framework is implemented and tested with Ubuntu 18.04, CUDA 11.3, Python 3.7, Pytorch 1.12.1, NVIDIA 1080 Ti GPU. Unless otherwise stated the below scripts and instructions assume working directory is the project root. 


## Setup

- **Cuda & Python**

    In this project we utilize Pytorch with Python 3.7, Cuda 11.3, and a few Anaconda packages. Please review and follow this [installation guide](setup.md). However, feel free to try alternative versions or modes of installation. 

- **Data**

    Download the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset. Then place a softlink (or the actual data) in  *M3D-RPN/data/kitti*. 

	```
    cd M3D-RPN
	ln -s /path/to/kitti data/kitti
	```

	Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage. 

    ```
    python data/kitti_split1/setup_split.py
    python data/kitti_split2/setup_split.py
    ```
    
    Next, build the KITTI devkit eval for each split.

	```
	sh data/kitti_split1/devkit/cpp/build.sh
	sh data/kitti_split2/devkit/cpp/build.sh
	```
    
    Lastly, build the nms modules
    
    ```
	cd lib/nms
	make
	```

## Training

We use [visdom](https://github.com/facebookresearch/visdom) for visualization and graphs. Optionally, start the server by command line
```
python -m visdom.server -port 8100 -readonly
```
The port can be customized in *scripts/config* files. The training monitor can be viewed at [http://localhost:8100](http://localhost:8100)

First train resnet50 baseline model with below command.

```
python scripts/train_rpn_3d.py
```

After training baseline, change the baseline weight directory to traing LSQ.

The LSQ model can be trained with below command.

```
python scripts/train_rpn_3d_q.py
```

## Testing

Testing requires paths to the configuration file and model weights, exposed variables near the top *scripts/test_rpn_3d_q.py*. To test a configuration and model, simply update the variables and run the test file as below. 

```
python scripts/test_rpn_3d_q.py 
```
