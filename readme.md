# HOMOGAN

##Experiments
#### Experiment 1
- Added normalization

#### Experiment 2
- Normalization

#### Experiment 3
- Normalization

#### Experiment 4
- Normalization

#### Experiment 5
- Normalization

#### Experiment 6
- Normalization

#### Experiment 7
- Normalization

#### Experiment 8
- Normalization

#### Experiment 9
- Normalization

#### Experiment 10
- Normalization

#### Experiment 11
- Normalization

#### Experiment 12
- Normalization


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



### Features

- Support Standard Markdown / CommonMark and GFM(GitHub Flavored Markdown);
- Full-featured: Real-time Preview, Image (cross-domain) upload, Preformatted text/Code blocks/Tables insert, Code fold, Search replace, Read only, Themes, Multi-languages, L18n, HTML entities, Code syntax highlighting...;
- Markdown Extras : Support ToC (Table of Contents), Emoji, Task lists, @Links...;
- Compatible with all major browsers (IE8+), compatible Zepto.js and iPad;
- Support identification, interpretation, fliter of the HTML tags;
- Support TeX (LaTeX expressions, Based on KaTeX), Flowchart and Sequence Diagram of Markdown extended syntax;
- Support AMD/CMD (Require.js & Sea.js) Module Loader, and Custom/define editor plugins;
