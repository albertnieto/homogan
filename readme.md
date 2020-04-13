# HomoGAN
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/anieto95/homogan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/anieto95/homogan/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/anieto95/homogan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/anieto95/homogan/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Training a GAN from scratch and improve it by experiments 

## Table of Contents

- [HomoGAN](#homogan)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Running experiments](#running-experiments)
  * [Dataset](#dataset)
  * [Glossary](#glossary)
  * [Experiments](#experiments)
    + [Experiment 1](#experiment-1)
    + [Experiment 2](#experiment-2)
    + [Experiment 3](#experiment-3)
    + [Experiment 4](#experiment-4)
    + [Experiment 5](#experiment-5)
    + [Experiment 6](#experiment-6)
    + [Experiment 7](#experiment-7)
    + [Experiment 8](#experiment-8)
    + [Experiment 9](#experiment-9)
    + [Experiment 10](#experiment-10)
    + [Experiment 11](#experiment-11)
    + [Experiment 12, 13, 14](#experiment-12-13-14)
    + [Experiment 15](#experiment-15)

## Installation
    $ git clone https://github.com/anieto95/homogan
    $ cd homogan/
    $ sudo pip3 install -r requirements.txt

## Running experiments

In order to train the model, parameters should be set in `config.json`. Once parameters are set, simply run `main.py`.
Nevertheless, older experiments can be run as well. They can be found in `docs/ExperimentXX`. Though parameters can't be changed, they can be tested by running `docs/ExperimentXX/main.py`.

## Dataset

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter.

For the whole project, images have been cropped and reduced to 128x128px. For the Experiment 16, images were preprocessed to delete the background.

## Glossary
* Generator (G).
* Discriminator (D).
* Fully Connected (FC).
* Fully Convolutional (FConv).
* Dropout. 
>At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed.
* Label Smoothing.
>Label smoothing is a regularization technique for classification problems to prevent the model from predicting the labels too confidently during training and generalizing poorly.
* Label Flipping.
>Label flipping is a training technique where one selectively manipulates the labels in order to make the model more robust against label noise.
* Batch Normalization.
>Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks
* Spectral Normalization.
> Spectral Normalization normalizes the spectral norm of the weight matrix W, where the spectral norm σ(W) that we use to regularize each layer is the largest singular value of W. In few words, simply replaces every weight W with W/σ(W).
* Gaussian Noise.
>Gaussian Noise is statistical noise having a probability density function equal to that of the normal distribution, which is also known as the Gaussian distribution. In other words, the values that the noise can take on are Gaussian-distributed

## Experiments
### Experiment 1
First approach, architecture based on DCGAN.

#### Results
|Hyperparameters|Observations|
|:---|:---|
|Trainning size = 10.000<br/>Trainning Epochs = 35<br/>Batch Size = 16|* Huge model, generator with over 9M parameters in G vs 400k in the D.<br/> * Slow trainning per epoch and high memory consumption.|

|Results GIF|
|:---:|
|![](docs/Experiment01/results/png_to_gif_Exp1.gif)|

### Experiment 2
Change from previous models: 
* Wrap G and D definition in classes.
* Add tensorboard loss tracing.

#### Results
Trainning size = 10.000\
Trainning Epochs = 25\
Batch Size = 16

__Observation__: 
|![](docs/Experiment02/Gen_Loss.png)|![](docs/Experiment02/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment02/results/png_to_gif_Exp2.gif)|

### Experiment 3
Change from previous models: 
* Creation of two independent classes for data importing and GAN architecture definition.
* Remove conv layers 1 and 2 from G (reduce number of parameters).
* Remove layer 2 (Conv, BatchNorm, LeackyReLU and Dropout) from D.
* Adde fake and real accuracy metric.

#### Results
Trainning size = 10.000\
Trainning Epochs = 34\
Batch Size = 16

__Observation__: 
|![](docs/Experiment03/Gen_Loss.png)|![](docs/Experiment03/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
|![](docs/Experiment03/fake_acc.png)|![](docs/Experiment03/real_acc.png)|
|Fake accuracy|Real accuracy|

|Results GIF|
|:---:|
|![](docs/Experiment03/results/png_to_gif_Exp3.gif)|

### Experiment 4
Change from previous models: 
* The two FC input layers of the G changed to FConv.
* Update restriction on the D -> D is not updated while G loss is >4.

#### Results
Trainning size = 10.000\
Trainning Epochs = 20\
Batch Size = 16

__Observation__: 
|![](docs/Experiment04/Gen_Loss.png)|![](docs/Experiment04/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
|![](docs/Experiment04/fake_acc.png)|![](docs/Experiment04/real_acc.png)|
|Fake accuracy|Real accuracy|

|Results GIF|
|:---:|
|![](docs/Experiment04/results/png_to_gif_Exp4.gif)|

### Experiment 5
Change from previous models: 
* Removed restriction on D update

#### Results
Trainning size = 10.000\
Trainning Epochs = 20\
Batch Size = 16

__Observation__: 
|![](docs/Experiment05/Gen_Loss.png)|![](docs/Experiment05/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
|![](docs/Experiment05/fake_acc.png)|![](docs/Experiment05/real_acc.png)|
|Fake accuracy|Real accuracy|

|Results GIF|
|:---:|
|![](docs/Experiment05/results/png_to_gif_Exp5.gif)|

### Experiment 6
Change from previous models: 
* Added label smoothing (0 -> {0-0.1} and 1 -> {0.9-1})
* Added label flipping on 5% of labels

#### Results
Trainning size = 10.000\
Trainning Epochs = 20\
Batch Size = 16

__Observation__: 
|![](docs/Experiment06/Gen_Loss.png)|![](docs/Experiment06/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
|![](docs/Experiment06/fake_acc.png)|![](docs/Experiment06/real_acc.png)|
|Fake accuracy|Real accuracy|

|Results GIF|
|:---:|
|![](docs/Experiment06/results/png_to_gif_Exp6.gif)|

### Experiment 7
Change from previous models: 
* Change model architecture.
* Remove BatchNorm layers
* Remove Label Smoothing and Label flip

#### Results
Trainning size = 10.000\
Trainning Epochs = 100\
Batch Size = 100

__Observation__: 
|![](docs/Experiment07/DiscLossFake.png)|![](docs/Experiment07/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment07/GenLoss.png)|
|Generator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment07/results/png_to_gif_Exp7.gif)|

### Experiment 8
Change from previous models: 
* Only male images
* Added label smoothing (0 -> {0-0.1} and 1 -> {0.9-1})
* Added label flipping on 5% of labels

#### Results
Trainning size = 22.000\
Trainning Epochs = 100\
Batch Size = 200

__Observation__: 
|![](docs/Experiment08/DiscLossFake.png)|![](docs/Experiment08/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment08/GenLoss.png)|
|Generator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment08/results/png_to_gif_Exp8.gif)|

### Experiment 9
Change from previous models: 
* Using male and female images at 50%
* Introduced training ratio G:D, set to 1:3 (traing D 3 times more than G)

#### Results
Trainning size = 10.000\
Trainning Epochs = 100\
Batch Size = 100

__Observation__: 
|![](docs/Experiment09/DiscLossFake.png)|![](docs/Experiment09/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment09/GenLoss.png)|
|Generator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment09/results/png_to_gif_Exp9.gif)|

### Experiment 10
Change from previous models: 
* Chenge architecture to introduce conditioning GAN
* Only 1 feature allowed for conditioning 
* Ratio G:D, set to 1:1

#### Results
Trainning size = 10.000\
Trainning Epochs = 100\
Batch Size = 100

__Observation__: 
|![](docs/Experiment10/DiscLossFake.png)|![](docs/Experiment10/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment10/GenLoss.png)|
|Generator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment10/results/png_to_gif_Exp10.gif)|

### Experiment 11
Change from previous models: 
* Introduced training ratio G:D, set to 1:3 (traing D 3 times more than G)

#### Results
Trainning size = 10.000\
Trainning Epochs = 100\
Batch Size = 100

__Observation__: 
|![](docs/Experiment11/DiscLossFake.png)|![](docs/Experiment11/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment11/GenLoss.png)|
|Generator Loss|

|Results GIF|
|:---:|
|![](docs/Experiment11/results/png_to_gif_Exp11.gif)|

### Experiment 12, 13, 14
Using Experiment 11 as base: 
* Introduced Spectral Normalization
* Training ratio G:D used:
    * 1:1
    * 1:3 
    * 1:5 

#### Results
* Trainning size = 10.000
* Trainning Epochs = 100\220\100
* Batch Size = 100

__Observation__: 

|Ratio 1:1|Ratio 1:3|Ratio 1:5|
|:---:|:---:|:---:|
|![](docs/Experiment12/DiscLossFake.png)|![](docs/Experiment13/DiscLossFake.png)|![](docs/Experiment14/DiscLossFake.png)|
|Discriminator Loss Fake Exp 12|Discriminator Loss Fake Exp 13|Discriminator Loss Fake Exp 14|
|![](docs/Experiment12/DiscLossReal.png)|![](docs/Experiment13/DiscLossReal.png)|![](docs/Experiment14/DiscLossReal.png)|
|Discriminator Loss Real Exp 12|Discriminator Loss Real Exp 13|Discriminator Loss Real Exp 14|
|![](docs/Experiment12/GenLoss.png)|![](docs/Experiment13/GenLoss.png)|![](docs/Experiment14/GenLoss.png)|
|Generator Loss Exp 12|Generator Loss Exp 13|Generator Loss Exp 14|

<details>
<summary>Result GIF Experiment 12</summary>
<br>
 <IMG SRC="docs/Experiment12/results/png_to_gif_Exp12.gif">
</details>

<details>
<summary>Result GIF Experiment 13</summary>
<br>
 <IMG SRC="docs/Experiment13/results/png_to_gif_Exp13.gif">
</details>

<details>
<summary>Result GIF Experiment 14</summary>
<br>
 <IMG SRC="docs/Experiment14/results/png_to_gif_Exp14.gif">
</details>

### Experiment 15
Using Experiment 11 as base: 
* Implementation of Multi-labeling 
* Labels:
    * Bald
    * Glasses
    * Beard

#### Results

|Hyperparameters|Observations|
|:---|:---|
|Trainning size = 9000<br/>Trainning Epochs = 100<br/>Batch Size = 100|* This model was trained to be able to produce faces with different labels<br/> * First Experiment using the new flexible software that allows to choose labels and other configs of the model easily.|


__Observation__: 
|![](docs/Experiment15/DiscLossFake.png)|![](docs/Experiment15/DiscLossReal.png)|
|:---:|:---:|
|Discriminator Loss Fake|Discriminator Loss Real|
|![](docs/Experiment15/GenLoss.png)|
|Generator Loss|

![](docs/Experiment15/results/png_to_gif_Exp15.gif)|
