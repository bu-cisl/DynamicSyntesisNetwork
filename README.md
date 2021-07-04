# DynamicSyntesisNetwork

Code for dyniamic synthesis network for large-scale 3D descattering, as presented in out publication:
W.Tahir, L. Tian, "Adaptive 3D descattering with a dynamic synthesis network
", arXiv (2021), 2107.00484
https://arxiv.org/abs/2107.00484


![alt text](https://ibb.co/FmQj3Zw)

**environment.yaml** lists dependencies used to run this code on an Nvidia RTX-8000 GPU.


# Training new model
* The script **Train_new_model/main_train.py** contains the code for training the network. 

* In order to train the model, the data can be downloaded from the following google drive link:
XYZ

* Download the 'data' folder from the above link and copy it to the 'Train_new_model' folder, such that it's path is **.../Train_new_model/data**

* To begin training, execute the script main_train.py as follows:

$ python main_train.py

# Using pretrained model for segmentation
* The folder **Test_trained_model** contains a pretrained model and a script which can use that pretrained model to descatter a test 3D backpropagation volume. 
 
* In order to perform descattering on a sample 3D backpropagation (not used in the training process), download the folder 'test_data' from the following google drive link:
XYZ

* Copy the downloaded 'test_data' folder to the 'Test_trained_model' folder, such that it's path is **'.../Test_trained_model/test_data'**. Not that the data in this folder has already been pre-processed using the method outlined in our paper.

* In the folder **Test_trained_model**, execute the script main_test.py with default configurations as follows:

$ python main_test.py

* The model will perform descattering on all test data and save results in the folder 'test_results'



