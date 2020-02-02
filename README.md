# Startup Prediction

This repository contains solution for HKUST-UBS 'China Startup Prediction' Project from Team (Andrew and Zhiyun).


## 1. How to use
#### 1.1 Setup
```
conda create -n startup_prediction python=3.6
source activate startup_prediction
pip install -r requirements.txt
```

#### 1.2 Training
To train and evaluate model, run:
```
cd src/model
python <INSERT FILENAME>
```
For example, to train and evaluate lightGBM model, run:
```
cd src/model
python multi_lightgbm.py 
```



## 2. Methodology
#### 2.1 Feature Engineering
Some features extracted are:
* Company overview
  - includes sector, management team, location 
* Funding event
  - includes normalized funding amount, funding round code, number of investors
* Investor
  - includes top investors 
* News
  - includes positive and negative sentiment of news

#### 2.2 Model
The models implemented are as follows:
- LightGBM
  - as baseline model 
- Temporal Convolutional Network (TCN)
  - with dilations, causal network and skip connections
  

  
## 3. Implementation
#### 3.1 Modules and Repository Structure
- Codes are separated into three main modules: `data_loader`,`model`,`data`. 
- Subections in `docs` specifically addresses the criterias of the evaluation rubics.
    - For `Design - quality of background research`, please refer to `resources.md` and `model_description.md`
    - For `Code - organization of code`, please refer to `preprocessing_file_organization.jpeg` and `file_description.md` 



## 4. Results
The following results are computed as precision, recall and f1-score for test set:
- LightGBM
  - precision: 0.7652
  - recall: 0.5787
  - f1-score: 0.6076
- Temporal Convolutional Network (TCN)
  - precision: 0.8495
  - recall: 0.8539
  - f1-score: 0.8517
  


## 5. Future Work
- Experiment with additional graph propagation layer (similar to Graph Convolutional Network[1]) added between TCN block for feature propagation
- Finetune TCN



## References
[1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks.”

