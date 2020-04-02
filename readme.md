# HomoGAN
Training a GAN from scratch and improve it by experiments 

## Table of Contents

[TOC]

## Installation
    $ git clone https://github.com/anieto95/homogan
    $ cd homogan/
    $ sudo pip3 install -r requirements.txt	- TODO

## Running experiments

Simply run the `main.py`

## Dataset

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter.

## Glossary
* Generator (G)
* Discriminator (D)
* Fully Connected (FC)
* Fully Convolutional (FConv)

## Experiments
### Experiment 1
- Added normalization

### Experiment 2
- Normalization

### Experiment 3
- Normalization

### Experiment 4
Change from previous models: 
* The two FC input layers of the G changed to FConv.
* Update restriction on the D -> D is not updated while G loss is >4.

#### Results
Trainning size = 10.000\
Trainning Epochs = 20\
Batch Size = 16\

__Observation__: 
|![](src/Experiment4/Gen_Loss.png)|![](src/Experiment4/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
* Fake accuracy grows rapidly at the first epochs, reaching the value 0.95, but then decreases with each epoch and tends to stabilize around 0.7
![](src/Experiment4/fake_acc.png)
* Real accuracy increases at the beginning, reaching the value 0.8, but then decreases with each epoch and tends to stabilize around 0.45
![](src/Experiment4/real_acc.png)

### Experiment 5
Change from previous models: 
* Removed restriction on D update

#### Results
Trainning size = 10.000\
Trainning Epochs = 20\
Batch Size = 16\

__Observation__: 
|![](src/Experiment5/Gen_Loss.png)|![](src/Experiment5/Disc_Loss.png)|
|:---:|:---:|
|Generator Loss|Discriminator Loss|
* Fake accuracy 
![](src/Experiment5/fake_acc.png)
* Real accuracy 
![](src/Experiment5/real_acc.png)

### Experiment 6
- Normalization

### Experiment 7
- Normalization

### Experiment 8
- Normalization

### Experiment 9
- Normalization

### Experiment 10
- Normalization

### Experiment 11
- Normalization

### Experiment 12
- Normalization
