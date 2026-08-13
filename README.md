# SMARTWASH 4 SMARTBRUSH

## 📁 Data

The `data/` directory contains all the datasets used for training and evaluating the smartwatch brushing detection models. It is organized into the following subfolders:

### `data/1`
contains data of each brushing region collected separately.

### `data/2`
new_train_setx.csv contains data from sessions of brushing with pause (time inconsistent for each pauses) in between regions, where x represents the session number. Each session follows a specific set of order.

_order of brushing: front buccal -> left buccal -> right buccal ->left occlusal -> right occlusal -> left lingual -> front lingual -> right lingual_

BUCCAL -> facing the cheecks 
LINGUAL -> part on the inside (facing tongue) 
OCCLUSAL -> biting area

Furthermore this folder also contains the manually segmented data of each session into each region, with the transition movement completely removed.

### `data/3`
new_train_setx.csv is the same data collected as in `data/2`. However different segmentation is done, in which **for each session**, the **number of data points** for each region is the **same**.


## ⌚ Smartwatch
Documentation on some of the explored options is [here](https://nusu-my.sharepoint.com/:w:/g/personal/e1093041_u_nus_edu/IQA80aixHrdtSKUHw5Ci7PQGAZ-BINweqd0xKDqK3OFHdyk?e=nNjV4Q)

ESP32-S3 Development Board with round LCD was initially chosen, however overheat on the LCD flex cable occured and eventually broke. A similar model with a proper protection is then used. The new and current smartwatch is using ESP32-S3 with a 6-axis IMU (QMI8658) built in. Link is [here](https://www.aliexpress.com/item/1005009920287161.html?spm=a2g0o.productlist.main.17.240fxpxNxpxNWI&algo_pvid=23ef4a7b-5007-4f95-8741-d87a0d13ddf4&algo_exp_id=23ef4a7b-5007-4f95-8741-d87a0d13ddf4-14&pdp_ext_f=%7B%22order%22%3A%2239%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21SGD%2139.91%2139.91%21%21%21207.32%21207.32%21%402101529317761839700308726ea460%2112000050582108663%21sea%21SG%216251212718%21X%211%210%21n_tag%3A-29919%3Bd%3A1f1ecc6f%3Bm03_new_user%3A-29895&curPageLogUid=HPFg8PMsMM3k&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005009920287161%7C_p_origin_prod%3A#nav-specification)

### ESP32 Code
The code used is available in `Code/esp32s3lcd_test/esp32s3lcd_test.ino`. What it does is it set the configuration of the IMU, then read the IMU data. Data is then sent through UDP, which is captured and save in csv file. UDP code is available in `udp_receive.py`. In order for this to work, need to make sure that both the smartwatch and laptop (or other device running `udp_receive.py`) is connected to the same network and UDP is sent through the same port.

## 💻 AI Model
Currently, model used is based on random forest. This is because comparison was done with knn (`knn.py`), cnn (`cnn.py`), and AT-LSTM based on Hygiea+ paper (`atlstm_hygiea.py`), and it was found out that random forest produced the highest accuracy. This was tested on 4 session set of data, with 3 sessions being the training set and 1 session being the test set, thus test set is completely unseen during training. Each session data are manually segmented, completely removing the transition phase, thus data used were all clean data. Modification were made, such as by using relative motion of the brushing instead of the fixed data point from the accelerometer and gyroscope, rotation invariant, training data smoothing to improve the accuracy of detection. Experimentation on this was under the files `multiple_session_train_...py`.

LLM was then incorporated. The idea was for LLM to call on the random forest model, and analyse the result from the random forest before making the final decision. LLM prompt was given that people often brush the adjacent teeth next, and so prediction is affected by this prompt. But this LLM prompt was then discarded as people often brush their teeth with no certain order.

Under the files `full_segmented_...py`, instead of manually segmenting all the data, which is not ideal for real world usage, test data is given the full session, and segmentation is done before prediction. Training data still uses the manually segmented clean data for better representation, thus transition phase for the test data is removed as much as possible. Here LLM was removed due to many prediction altered by LLM changes the prediction from correct to wrong prediction. In improving the accuracy, more features are added, PCA (principal component analysis) and decision based on probability voting is used, and random forest configuration is tuned.