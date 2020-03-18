# HOMOGAN

## Glossary
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
_Observation_: 
* Fake accuracy grows rapidly at the first epochs, reaching the value 0.95, but then decreases with each epoch and tends to stabilize around 0.7
![](src/Experiment4/fake_acc.png)
* Real accuracy increases at the beginning, reaching the value 0.8, but then decreases with each epoch and tends to stabilize around 0.45
