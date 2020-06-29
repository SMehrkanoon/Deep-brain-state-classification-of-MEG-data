# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                 #
#  **Please make sure you have the next dependences installed**   #
#                                                                 #
#  pip install NumPy                                              #
#  pip install Scipy                                              #
#  pip install Matplotlib                                         #
#  pip install PySurfer                                           #
#  pip install Scikit-learn                                       #
#  pip install Numba                                              #
#  pip install NiBabel                                            #
#  pip install Pandas                                             #
#  pip install DIPY                                               #
#  pip install PyVista                                            #
#  pip install pyriemann                                          #
#  pip install mne                                                #
#  pip install re                                                 #
#  pip tensorflow-gpu                                             #
#  pip install boto3                                              #
#                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                 #
#  **MAKE SURE THE MODEL IS CREATED, SAVED AND                    #
#    LOAD WITH THE SAME VERSION OF TENSORFLOW                     #
#                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import tensorflow as tf

if tf.__version__=='1.15.0':
  tf.enable_eager_execution()

import os.path as op
import os
import data_utils_EEGNet as utils
import boto3
import mne
import h5py
from mne import io
from mne.datasets import sample
import numpy as np
import sys
from keras import utils as np_utils
import gc
from sklearn.model_selection import train_test_split
import re
from EEGModelSelfGlobalAttention import EEGNetAttention


####################### CREATING DATASET #######################

list_subjects = ['105923', '164636', '133019',
                 '113922','116726','140117',
                 '175237', '177746', '185442',
                 '191033', '191437', '192641',
                 '204521','212318', '162935']
           

n_subjects_test = 3  #Number of subjects used for the test (It will take the n last ones)
n_subjects_stage = 3 #Number of subjects load in memory in each stage of the training


#Split train/test
list_subjects_train = list_subjects[:-n_subjects_test]
list_subjects_test = list_subjects[-n_subjects_test:]


#Spliting the traning subjects in different stages
subjects_stages_train = []
for i in range(0, len(list_subjects_train), n_subjects_stage):
  subjects_stages_train.append(list_subjects_train[i:i+n_subjects_stage])

#Creating directory to save models
os.mkdir("saved_models")


####################### CREATING MODEL #######################

#Properties of the model
kernel_lenght = 128   
temporal_filter = 16   
spatial_filter = 2    
pointwise_filter = 32 #(Recommended value in EEGNet - ()temporal_filter * spatial_filter)
dropout_rate = 0.3    #value for switching off neurons
classes_classification = 4 #FIXED

#Attention mechanism properties
depth_k=6
depth_v=4
num_heads=2

#Number channels
n_chans=248 #FIXED
#Number time points
samples = 1425 #FIXED
#Kernels (To transform data to NCHW (batch_size, channel, height, width)
kernels = 1


#Properties for the compiling
loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']


#EEGNet with attention
model = EEGNetAttention(nb_classes = classes_classification, Chans = n_chans, Samples = samples, 
                  dropoutRate = dropout_rate, kernLength = kernel_lenght, F1 = temporal_filter, 
                  D = spatial_filter, F2 = pointwise_filter, 
                  dropoutType = 'Dropout', depth_k=depth_k, depth_v=depth_v, num_heads=num_heads)

model.compile(loss=loss, optimizer=optimizer, metrics = metrics)
#Printing model
print("\n\n>>The model about to be trained is the next one:\n",model.summary())




####################### TRAINING MODEL #######################

#Properties for the traning
batch_size = 16
n_epochs = 50
verbose = 1


#Training loop 
for n in range(n_epochs):
  print("\n\n>>>>>>>>   || Epoch",n+1,"||   <<<<<<<<\n")
  for subject in list_subjects_train:
    print("-- Training on subject", subject)

    #Reading data from the subject
    with h5py.File("datasets/"+subject+".h5","r") as hf:
      X_train = np.array(hf.get("x_train_"+subject))
      Y_train = np.array(hf.get("y_train_"+subject))
      X_validate = np.array(hf.get("x_validate_"+subject))
      Y_validate = np.array(hf.get("y_validate_"+subject))

      #Fitting the model with one subject and one epoch
      model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, 
                            verbose = verbose, validation_data=(X_validate, Y_validate), 
                            callbacks=None)
      #Freeing memory
      del X_train
      del Y_train
      del X_validate
      del Y_validate
      gc.collect()

  ####### Test each 5 epochs CROSS-SUBJECT #######
  if n%5==0 and n>0:
    #Creating dataset for testing
    for subject in list_subjects_test:
      print("-Reading data from subject", subject)
      with h5py.File("datasets/"+subject+".h5","r") as hf:
        if 'X_test' not in globals():
          X_test = np.array(hf.get("x_train_"+subject))
          Y_test = np.array(hf.get("y_train_"+subject))
          X_test = np.concatenate((X_test, np.array(hf.get("x_validate_"+subject))))
          Y_test = np.concatenate((Y_test, np.array(hf.get("y_validate_"+subject))))
        else:
          X_test = np.concatenate((X_test, np.array(hf.get("x_train_"+subject)), np.array(hf.get("x_validate_"+subject))))   
          Y_test = np.concatenate((Y_test, np.array(hf.get("y_train_"+subject)), np.array(hf.get("y_validate_"+subject))))   

    #Evaluation of the model 
    print("\n\n>>Evaluation CROSS-SUBJECT: ")
    result = model.evaluate(X_test, Y_test, batch_size = 16)
    print(result)
    #Freeing memory
    del X_test
    del Y_test
    #Saving the model in local
    tf.saved_model.save(model, 'saved_models/model_EEGNet'+str(n))
    
#Saving last model
tf.saved_model.save(model, 'saved_models/model_EEGNet_Last')

 