
# Implementation of MICCAI 2019 paper

This repository is a re-implementation of our miccai 2019 paper: [Pick-and-Learn: Automatic Quality Evaluation for Noisy-Labeled Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_64). Note that I cannot get access to the original code since I just graduated this year, this re-implementation is not exactly the same as the version for MICCAI submission:

* Data split is 162 samples for training and 85 for testing.
* Segmentation module is a 'U-Net' like network instead of the real U-Net. Also, the Quality Awareness module is just the first half of the U-Net with global average pooling instead of VGG mentioned in the paper.

----------------------------

## Installation

* Clone this repository
* Create an environment with `envs/environment.yml`
* Run `CUDA_VISIBLE_DEVICES=$GPU_ID python train.py`

## Training

### Command

* Activate previously created conda environment : `source activate ins-seg-pytorch`.
* Run `train.py`.

```
usage: train.py [-h] [-o OUTPUT] [-mi MAX_ITER] [-lr LR]
                [--iter-save SAVE_PER_ITERATIONS] [-n NOISE_PERCENT] 
		[-ns NOISE_RANGE_MIN] [-nm NOISE_RANGE_MAX]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE]

Quality Awareness Model

optional arguments:
  -h, --help                Show this help message and exit
  -o, --output              Output path
  -mi, --max-iter           Maximum iterations
  -n, --noise               Noise percentage
  -ns, --max-iter           Noise range (min)
  -nm, --max-iter           Noise range (max)
  -lr                       Learning rate [Default: 0.0001]
  --iter-save               Number of iterations to save
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
```
