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
from EEGModelAttention import EEGNetAttention




####################### SUBJECTS TO LOAD #######################

# Uncomment the correct one! Depending of whether the model 
# uses attention or not

#model = load_model('model_EEGNet.h5') #No attention
model = tf.keras.models.load_model('saved_models/model_EEGNet_Last') #Attention


####################### SUBJECTS TO LOAD #######################
list_subjects_test = ['204521','212318', '162935',
                      '601127','725751','735148'] 


n_subjects_stage = 3 #Number of subjects load in memory in each stage of the training


#Spliting the testing subjects in different stages
subjects_stages_test = []
for i in range(0, len(list_subjects_test), n_subjects_stage):
  subjects_stages_test.append(list_subjects_test[i:i+n_subjects_stage])



####################### TESTING LOOP #######################
for i in range(len(subjects_stages_test)):


  ####################### FIRST TEST - Whole recording #######################

  #Reading data, extracting epochs and creating data-set
  print("\n\n\n>>Creating a dataset with the subjects: ",subjects_stages_test[i],"\n\n\n")
  X_complete, Y_complete, pick_chans = utils.create_dataset(subjects_stages_test[i])

  #Convert labels to one-hot encodings (00010000000...)
  Y_complete      = np_utils.to_categorical(Y_complete)

  #Number channels (Same for task and resting)
  n_chans = len(pick_chans)
  #Number of time points in each sample (Same for task and resting)
  samples = X_complete.shape[-1]
  #Kernels (To transform data to NCHW (batch_size, channel, height, width)
  kernels = 1

  #Convert data to NCHW (trials, kernels, channels, samples)
  X_complete      = X_complete.reshape(X_complete.shape[0], kernels, n_chans, samples)

  
  #Evaluation of the model with the test data set
  print("\n\n>>Evaluation after stage",i)
  result = model.evaluate(X_complete, Y_complete, batch_size = 16)
  print(result)

  #Writing the file
  filename = "Experiments/EEGNet/Tests_summary.txt"
  with open(filename, "a+") as file:
    	file.write("\n\n\n>>Tested over the next list of subjects: \n  - "+str(subjects_stages_test[i])+"\n  - Result -> loss: "+str(result[0])+" - acc: "+str(result[1]))

  #Freeing memory
  del X_complete
  del Y_complete
  
  ####################### SECOND TEST - removing beginning recording #######################

  #Extracting "chunks" different tasks
  print("\n\n\n>>Creating a dataset with the subjects: ",subjects_stages_test[i],"\n\n\n")
  X_complete, Y_complete, pick_chans = utils.create_dataset_cut(subjects_stages_test[i])

  #Convert labels to one-hot encodings (00010000000...)
  Y_complete      = np_utils.to_categorical(Y_complete)

  #Number channels (Same for task and resting)
  n_chans = len(pick_chans)
  #Number of time points in each sample (Same for task and resting)
  samples = X_complete.shape[-1]
  #Kernels (To transform data to NCHW (batch_size, channel, height, width)
  kernels = 1

  #Convert data to NCHW (trials, kernels, channels, samples)
  X_complete      = X_complete.reshape(X_complete.shape[0], kernels, n_chans, samples)

  #Evaluation of the model with the test data set
  print("\n\n>>Evaluation after stage",i)
  result = model.evaluate(X_complete, Y_complete, batch_size = 16)
  print(result)

  #Writing the file
  filename = "Experiments/EEGNet/Tests_summary.txt"
  with open(filename, "a+") as file:
    	file.write("\n\n\n>>(CUT) Tested over the next list of subjects: \n  - "+str(subjects_stages_test[i])+"\n  - Result -> loss: "+str(result[0])+" - acc: "+str(result[1]))

 
  #Freeing memory
  del X_complete
  del Y_complete

