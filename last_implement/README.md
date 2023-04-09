Code adapted from [lachinov/brats2018-graphlabunn (github.com)](https://github.com/lachinov/brats2018-graphlabunn)

Code partially changed



**Where to put the dataset**

Unzip the dataset on Kaggle ([BRATS-2018 | Kaggle](https://www.kaggle.com/datasets/sanglequang/brats2018))

Put the training data and valid data in the right folder

*HGG* and *LGG* should be the subfolder of the *training_data*



**To run the data augmentation (as well as the data pre-processing)**

First make sure that *training_data* has no *numpy_dataset* subfolder.

Run the augment_dataset.py
