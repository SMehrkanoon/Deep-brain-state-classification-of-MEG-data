import data_utils_EEGNet as utils
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from keras import utils as np_utils
import h5py

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# 																	                                  #
# The script downloads the subjects and saves the subject in          #
# both the original format and h5 format (preprocessing them as well) #
# ready to use with EEGNet                                            #
#                                                                     #
# To download the subjects from Amazon S3 server you need an account  #
# on such a server and enough disk memory as the are quite heavy      #
#																	                                    #
# ------------------------------------------------------------------- #
#																  	                                  #	
# FILL THE LINES WITH THE PERSONAL & SECRET ACCESS KEY BELOW!! :)	    #
#																	                                    #
# ------------------------------------------------------------------- #
#																  	   																#
#  **Please make sure you have the next dependences installed**       #
#																  	                                  #
#  pip install NumPy									                       		      #
#  pip install boto3										                       	      #
#  pip install sklearn                                                #
#  pip install keras                                                  #
#  pip install h5py                                                   #
#																                                      #  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#List of the subjects to be dowloaded
list_subjects = ['105923','164636', '133019',
                 '113922','116726','140117',
                 '175237', '177746', '185442',
                 '191033', '191437', '192641',
                 '601127','725751','735148',
                 '204521','212318', '162935']

           
#Amazon S3 credentials
personal_access_key_id = 'XXXXXXXXXX'
secret_access_key = 'XXXXXXXXXX'



#Creating a directory for the downloaded files
try:
	os.mkdir('subjects')
except:
	pass
try:
	os.mkdir('datasets')
except:
	pass

#Downloading subjects
for subject in list_subjects:
  utils.download_subject(subject,personal_access_key_id,secret_access_key)

  #Reading data, extracting epochs and creating data-set
  print("\n\n\n>>Creating a dataset with the subject: ",subject,"\n\n\n")
  try:
    #Creating epochs and preprocessing 
  	X_train, Y_train, pick_chans = utils.create_dataset([subject])
  except:
  	print("\n\nERROR! \nProbably your Amazon S3 credentials are not valid...\n\n")
  	sys.exit()

  #Splitting
  X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.2, random_state=1) #Second split the validation (20%)


  #Convert labels to one-hot encodings (00010000000...)
  Y_train      = np_utils.to_categorical(Y_train)
  Y_validate   = np_utils.to_categorical(Y_validate)


  #Number channels (Same for task and resting)
  n_chans = len(pick_chans)
  #Number of time points in each sample (Same for task and resting)
  samples = X_train.shape[-1]
  #Kernels (To transform data to NCHW (batch_size, channel, height, width)
  kernels = 1


  #Convert data to NCHW (trials, kernels, channels, samples)
  X_train      = X_train.reshape(X_train.shape[0], kernels, n_chans, samples)
  X_validate   = X_validate.reshape(X_validate.shape[0], kernels, n_chans, samples)
  

  #Storing the dataset in h5 format
  destination_file = "datasets/"+str(subject)+".h5"
  with h5py.File(destination_file,"w") as hf:
      hf.create_dataset("x_train"+'_'+str(subject), data=X_train,compression="gzip", compression_opts=4)
      hf.create_dataset("x_validate"+'_'+str(subject), data=X_validate,compression="gzip", compression_opts=4)
      hf.create_dataset("y_train"+'_'+str(subject), data=Y_train,compression="gzip", compression_opts=4)
      hf.create_dataset("y_validate"+'_'+str(subject), data=Y_validate,compression="gzip", compression_opts=4)

  #Freeing memory
  del X_train
  del Y_train
  del X_validate
  del Y_validate

