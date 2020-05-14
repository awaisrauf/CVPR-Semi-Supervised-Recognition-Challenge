## Overall Approach 


## Augmentation 

During experiments, I observed that augmentation based training requrie more time to converge. From [1]:
> We found that regularization methods including Stochastic Depth [17], Cutout [3], Mixup [48], and CutMix require a
greater number of training epochs till convergence. Therefore, we have trained all the models for 300 epochs with
initial learning rate 0.1 decayed by factor 0.1 at epochs
75, 150, and 225. The batch size is set to 256. The hyperparameter α is set to 1. We report the best performances of
CutMix and other baselines during training




## References 
1. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
