# ML4HCProject2

This repository contains different natural language processing models for (sequentially) classifying sentences in abstracts of 
Randomized Control Trials (RCTs). Our models are trained and tested on the [PubMed RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct) 
For a detailed discussion of the models and their performances we refer to the provided report. 
We implemented the following sentence embeddings and corresponding models for classification

1. TF-IDF embedding: The final classifier is a logistic regression. (task 1)
2. Word2Vec embedding: The best performing classifier is Bidrectional LSTM + fully connected layers (task 2)
3. BERT embedding: 
4. Integrating structual context: We first use sentence embeddings from task 2 and 3, respectively. Then, 
   the final classifier is a bidirectional LSTM with a neural network classifier on top (hierarchical abstract model),
5. Knowledige Distillation: We apply knowledge distillation with hierarchical abstract model + BERT as teacher and 
   models from task 1 and 2 as students. 

## Getting started
We use a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment for dependency management. 
Therefore the minimal requirements are Conda. To create an environment, open an Anaconda prompt and run the following:
```
conda env create -f project2_environment.yml
```
If antivirus related permission issues or os access errors occur, running the Anaconda prompt on administrator mode avoids such issues.

## Add your files
The data input files should be added into a new folder `data/`. Calling the function ```download_data``` from the project2Lib-package
automatically downloads the PubMed RCT dataset and places it in the corresponding folder.

## Project2Lib

This package includes functions that were used for training and testing our models and are called from the respective notebooks.

To use the library in your code run, e.g.

```
import project2Lib

data = project2Lib.load_data_as_dataframe()

```

In ```BERT``` we included all relevant functions to working with BERT.
```Data``` has different functions for dowloading, loading and preprocessing the data and obtaining sentence embedders. In ```Sequence``` contains helpers for creating hierarchical models
```KD``` has utilities for knowledge distilation.
Last, in ```Utilities``` we included the metrics used. 

## Testing and evaluation of different models
The code that we use for evaluating different models is in the respective notebooks. Further the code for the final
models are also included. The following gives a brief overview over the corresponding notebooks and their contents.
To reproduce our results from the report, follow the exact order of the tasks and their corresponding scripts
as they are described in the following.

### Preprocessing 
For preprocessing the data, the notebook **Preprocessing.ipynb** must be run in the exact order as cells occur. 
It creates different preprocessed versions that are used for later models.

### TF-IDF
For the steps taken to implement and evaluate the baseline model, which uses a TF-IDF embedding, refer to **TFIDF_BaelineModel.ipynb** 
and run all cells in the exact order cells occur.

### Word2Vec
For the steps taken to develop the models based on Word2Vec embeddings, refer to the following files. 
For a demonstration of model performance, all listed notebooks  must be run in the exact order that cells occur.

| Code | Description |
| -------------- | --------- |
| Word2Vec_Embedding_Generation.ipynb | Embedding generation, semantic realtionships and visualisations  | 
| Word2Vec_Averaged_Embedding_Approach.ipynb | Non-sequential classifiers using averaged sentence vectors| 
| Word2Vec_Sequential_Classifier.ipynb | Sequential classifiers using word vectors | 


### BERT

| Code | Description |
| -------------- | --------- |
| TrainingBERT.ipynb | Training BERT-models. The user has to manualy select which model to run | 

### Integrating Structure

| Code | Description |
| -------------- | --------- |
| TrainingHiercical.ipynb | Training hierarchical models. | 
| TrainingKD.ipynb | Distilling knowledge into TF-IDF model. | 

#### 



## Authors
Mert Ertugrul <br>
Johan Lokna <br>
Nora Schneider