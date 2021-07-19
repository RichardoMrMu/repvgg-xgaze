# repvgg-xgaze
[Challenge](https://competitions.codalab.org/competitions/28930)
Get the 8th place in the ETH-XGaze Competition and use repvgg. Get 3.96 error degrees.
HUST-IETC method for GAZE 2021 competition on the ETH-XGaze dataset

## Environment Setup
The setup is similar to that of the baseline method.

We have tested this code-base in the following environments:
* Ubuntu 18.04
* Python 3.7 
* PyTorch 1.5.1

Clone this repository somewhere with:

    git clone https://github.com/RichardoMrMu/repvgg-xgaze.git
    cd repvgg-xgaze/

Then from the base directory of this repository, install all dependencies with:

    pip install -r requirements.txt


## Usage

### Directories specification

1. You should specify the dataset path in './config/repvgg_d2se_train.yaml' and if you have the pretrain model, you should set the weights your pretrain model path.

2. Please set the ckpt_dir path.


### Running evaluation
For  evaluation results use the baseline code.


## Citation
If you find this post helpful, please cite:


For GAZE 2021 competition on EVE dataset, please cite:

    @inproceedings{Park2020ECCV,
      author    = {Seonwook Park and Emre Aksan and Xucong Zhang and Otmar Hilliges},
      title     = {Towards End-to-end Video-based Eye-Tracking},
      year      = {2020},
      booktitle = {European Conference on Computer Vision (ECCV)}
    }

