# SMARTWASH 4 SMARTBRUSH

## 📁 Data

The `data/` directory contains all the datasets used for training and evaluating the smartwatch brushing detection models. It is organized into the following subfolders:

### `data/1`
contains data of each brushing region collected separately.

### `data/2`
new_train_setx.csv contains data from sessions of brushing with pause (time inconsistent for each pauses) in between regions, where x represents the session number. Each session follows a specific set of order.

order of brushing: front buccal -> left buccal -> right buccal ->left occlusal -> right occlusal -> left lingual -> front lingual -> right lingual 

Furthermore this folder also contains the manually segmented data of each session into each region, with the transition movement completely removed.

### `data/3`
new_train_setx.csv is the same data collected as in `data/2`. However different segmentation is done, in which *for each session*, the *number of data points* for each region is the *same*.