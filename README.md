# Semi-Supervised Recognition Challenge - FGVC7
This project contains my code for [CVPR2020]() challenge on Semi-Supervised Recognition.

- [Kaggle Challenge Link](https://www.kaggle.com/c/semi-inat-2020)
- [Github Link](https://github.com/cvl-umass/semi-inat-2020)

![](imgs/banner.png)

## Description 
This challenge is focused on learning from partially labeled data, a form of semi-supervised learning. This dataset is designed to expose some of the challenges encountered in a realistic setting, such as the fine-grained similarity between classes, significant class imbalance, and domain mismatch between the labeled and unlabeled data.

###  Major Components of the Project
1. Transfer Learning (ImageNet --> ..)
2. Fine Grained Classification
3. Long Tail Classification 
4. Semi Supervised Learning (A huge unlabelled dataset)
5. Learning From out of distribution data

### Data
This challenge focusses on Aves (birds) classification where we
provide labeled data of the target classes and unlabeled data from
target and non-target classes.
The data is obtained from iNaturalist, a community
driven project aimed at collecting observations of biodiversity.

The dataset comes with standard training, validation and test sets.
The training set consists of:

* **labeled images** from 200 species of
Aves (birds), where 10% of the images are labeled.
* **unlabeled images** from the same set of classes
as the labeled images (**in-class**).
* **unlabeled images** from a different set of classes as the
  labeled set (**out-of-class**). 
  These images are from a different set of classes in the Aves taxa.
  This reflects a common scenario where a coarser taxonomic label of
  an image can be easily obtained.
  
The validation and test set contain 10 and 20
images respectively for each of the 200 categories in the labeled set.
The distributions of these images are shown in the table below.

| Split | Details | Classes	| Images |
|:------:|:-------:|:--------:|:-------------:|
Train | Labeled | 200 |3,959|
Train | Unlabeled, in-class | 200 |26,640|
Train | Unlabeled, out-of-class | - |122,208|
Val  | Labeled | 200 | 2,000|
Test | Public | 200 |4,000|
Test | Private| 200 |4,000|

The number of images per class follows a heavy-tailed distribution as
shown in the Figure below.

![Train Val Distribution](imgs/class_dist1.png)

### Exploring Data

#### Class Distribution
![](imgs/class_dist.png)

#### train and validation
![](imgs/train_val_sample.png)

#### test

![](imgs/test_sample.png)

#### in-distribution data

![](imgs/in_dist_sample.png)
#### out-distribution data
![](imgs/out_dsit_sample.png)



## Overview of Approach 
Primarily, I will tackle the problem as transfer learning and use best practices from 5 different domains to further increase accuracy. 


### ToDos
- [ ] Transfer Learning 

### Coding ToDos
-[ ] Data
    - [ ] DataLoader
        -[ ] Custom DataLoader for 
        -[ ] 
    - [x] Data Augmentation: Create a file to augment data
        - [ ] Balanced sampler
        - [ ] How to use augmentation for better results (augmentation is not working as well as i was expecting.)
-[ ] Models
    -[x] Create getModels
-[ ] Hyperparameter Tuning
    
- [ ] Cycle Learning Rate
- [ ] Class Imbalance
- [ ] Noisy Teacher: with diff batch norm at the end on clean data?

### File Description 
- final_train.py: 
- train: ..


## References 
1. https://github.com/victoresque/pytorch-template
2. https://github.com/cvl-umass/semi-inat-2020

