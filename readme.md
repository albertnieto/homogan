# HOMOGAN

### Glossary
* Generator (G)
* Discriminator (D)
* Fully Connected (FC)
* Fully Convolutional (FConv)

## Experiment 4
Change from previous models: 
* The two FC input layers of the G changed to FConv.
* Update restriction on the D -> D is not updated while G loss is >4.

### Results:
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

## Experiment 5
Change from previous models: 
* Removed restriction on D update

### Results:
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
