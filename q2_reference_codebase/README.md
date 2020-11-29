# End-to-end-Sequence-Labeling-via-CNNs-CRF 

This repository is built upon this [**End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF**](https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb) codebase and seeks to replace the LSTM-based word-level encoder with a CNN layer. 


## Setup 

### 1. Upload the entire **q2_reference_codebase** folder into google drive. 
This is to utilise the GPU provided in Google Colabs for the training of models. 


The **q2_reference_codebase** folder should contain 

#### 1. /data directory which contains the dataset for this repository
- eng.testa
- eng.testb
- eng.train
#### 2. /models directory which contains pre-trained models 
- CNN-1-layer
- CNN-1-layer--relu
- CNN-2-layer
- CNN-2-layer--relu
- CNN-3-layer
- CNN-3-layer--relu
#### 3. **Named_Entity_Recognition-Word CNN-CRF.ipynb** notebook 


### 2. Download GloVe vectors and extract glove.6B.100d.txt into "./data/" folder
`wget http://nlp.stanford.edu/data/glove.6B.zip`

### 3. Installation
The best way to install pytorch is via the [**pytorch webpage**](http://pytorch.org/)

####  PyTorch Installation command:
`conda install pytorch torchvision -c pytorch`

####  NumPy installation
`conda install -c anaconda numpy`

### 4. Open **Named_Entity_Recognition-Word CNN-CRF.ipynb** in Google Colab

#### **Important things to note**:
1. Google drive must be mounted and path to **q2_reference_codebase** has to be navigated to<br>
    This codeblock should only be run once and can be commented out after. 

    `from google.colab import drive`<br> 
    `drive.mount('/content/drive')`<br>
    `%cd 'drive/*path to q2_reference_codebase*'`
2. Pre-trained models have been included in /models directory and can be loaded in to test testdata set.
3. In the first code cell, edit the parameters['layers'] and parameters['relu'] in order to choose your model,
and uncomment out parameters['reload']=False in order to train the model you have chosen.  
