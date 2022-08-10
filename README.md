
## Non-Smooth Energy Dissipating Networks

This repository is the official implementation of [Non-Smooth Energy Dissipating Networks]( ).

![Here comes the image](./denoising.png?raw=true "")

### Abstract
Over the past decade, deep neural networks have been shown to perform extremely well on a variety of image reconstruction tasks. Such networks do, however, fail to provide guarantees about these predictions, making them difficult to use in safety-critical applications. Recent works addressed this problem by combining model- and learning-based approaches, e.g., by forcing networks to iteratively minimize a model-based cost function via the prediction of suitable descent directions.
While previous approaches were limited to continuously differentiable cost functions, this paper discusses a way to remove the restriction of differentiability. We propose to use the Moreau-Yosida regularization of such costs to make the framework of energy dissipating networks applicable.  We demonstrate our framework on two exemplary applications, i.e., safeguarding energy dissipating denoising networks to the expected distribution of the noise as well as enforcing binary constraints on bar-code deblurring networks to improve their respective performances.  



## Get Started

- **Dependencies** \
-- install PyTorch (e.g. version 1.10.1+cu111) and Torchvision (e.g. version 0.11.2+cu111)
-- install requirements by: `pip install -r requirements`
- **Data** \
-- please configure the desired paths for the datasets in *path.json*
-- run `python gen_data.py` to generate barcodes and load the images used for denoising
- **Training and Evaluation** \
-- to train the networks, run `train_deblurring.sh` or `train_deblurring.sh` .
-- evaluate the trained networks in *evaluate_deblurring.ipynb* and *evaluate_denoising.ipynb*
-- for some tests on pretrained networks, load [those networks](https://drive.google.com/drive/folders/1WQ_8HSFYS0TAWtHEjCmnh1Tg5avQZcIv?usp=sharing) into *./examples/*
