
import mne
from mne import io
from mne.datasets import sample
import numpy as np
import os
import sys
import gc
from sklearn.model_selection import train_test_split
import boto3
import os.path as op
import os
import re
import h5py



#Download given subjects (ONLY DIRECTORIES GIVEN -> HAVE TO CHANGE IT)
def download_subject(subject,personal_access_key_id,secret_access_key):
  print("\n\n>>Downloading subject", subject,"\n")
  s3 = boto3.client('s3', aws_access_key_id=personal_access_key_id, aws_secret_access_key=secret_access_key)
  folders = ["3-Restin","6-Wrkmem", "8-StoryM", "10-Motort"]
  filenames = ["c,rfDC", "config"]
  for filename in filenames:
    for folder in folders:
      if filename == "c,rfDC":
        print("downloading c,rfDC file from folder {} ...".format(folder))
        print()
      if(op.exists(os.getcwd()+filename)):
        print("File already exists, moving on ...")
        print()
        pass
      try:
        if folder==folders[0]:
          filename_temp = filename+'_'+subject+'_rest'
        if folder==folders[1]:
          filename_temp = filename+'_'+subject+'_mem'
        if folder==folders[2]:
          filename_temp = filename+'_'+subject+'_math'
        if folder==folders[3]:
          filename_temp = filename+'_'+subject+'_motor'
        s3.download_file('hcp-openaccess', 'HCP_1200/'+subject+'/unprocessed/MEG/'+folder+'/4D/'+filename, "subjects/"+filename_temp)
      except Exception as e:
        print()
        print("the folder '{}' for subject '{}' does not exist in Amazon server, moving to next folder ...".format(folder,subject))
        print("Exception error message: "+str(e))
        pass


#Deletes the subjects given from the disk
def delete_subjects(list_subjects):
	for subject in list_subjects:
	  os.remove(os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_rest')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/config_'+subject+'_rest')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_mem')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/config_'+subject+'_mem')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_math')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/config_'+subject+'_math')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_motor')) 
	  os.remove(os.path.join(sys.path[0], 'subjects/config_'+subject+'_motor')) 



#Save the data of a subject in h5 format
def save_dataset_subject(X_train, Y_train, pick_chans, subject):
	hf = h5py.File('datasets/subject_'+subject)
	hf.create_dataset(subject+'_x', data=X_train)
	hf.create_dataset(subject+'_y', data=Y_train)
	hf.create_dataset(subject+'_chan', data=pick_chans)
	hf.close()



#Load an already exsisting dataset with the subjects given
def load_dataset(list_subjects):
	#To concatenate the X & Y between patients
	first_time = True
	#Iterating among subjects
	for subject in list_subjects:
	  print("\n\n>>LOADING DATA FROM SUBJECT",subject, "...\n\n")
	  hf = h5py.File('datasets/subject_'+subject)
	  if first_time:
	    X_train = hf.get(subject+'_x')
	    Y_train = hf.get(subject+'_y')
	    pick_chans = hf.get(subject+'_chan')
	    first_time = False
	  else:
	  	X_train2 = hf.get(subject+'_x')
	  	Y_train2 = hf.get(subject+'_y')
	  	X_train = np.concatenate((X_train, X_train2))
	  	Y_train = np.concatenate((Y_train, Y_train2))
	return X_train, Y_train, pick_chans


# Reads the data from the given subjects and creates
# a dataset with that data (return X_train & Y_train)
def create_dataset(list_subjects):
	#To concatenate the X & Y between patients
	first_time = True
	#Iterating among subjects
	for subject in list_subjects:
	  print("\n\n>>READING DATA FROM SUBJECT",subject, "...\n\n")
	  #Paths to the files
	  file_raw_rest =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_rest') #From real data
	  config_rest =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_rest') #From real data
	  file_raw_mem =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_mem') #From real data
	  config_mem =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_mem') #From real data 
	  file_raw_math =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_math') #From real data
	  config_math =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_math') #From real data
	  file_raw_motor =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_motor') #From real data
	  config_motor =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_motor') #From real data
	  #Reading resting data
	  raw_rest = mne.io.read_raw_bti(file_raw_rest, config_rest, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading memory data
	  raw_mem = mne.io.read_raw_bti(file_raw_mem, config_mem, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading math data
	  raw_math = mne.io.read_raw_bti(file_raw_math, config_math, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading motor data
	  raw_motor = mne.io.read_raw_bti(file_raw_motor, config_motor, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)

	  ####################### EXTRACTING EVENTS AND EPOCHS (second way -> Same size in the "chunks")#######################

	  #Creating "fake" events for the rest data (to extract epochs)
	  events_rest = mne.make_fixed_length_events(raw_rest, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_rest = mne.pick_types(raw_rest.info, meg=True, eeg=False, stim=False, eog=False)
	  picks = picks_rest
	  #Generating epochs
	  epochs_rest = mne.Epochs(raw_rest, events_rest, event_id = None, proj=False, picks=picks_rest, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs rest: ",len(epochs_rest.get_data()))
	  #Freeing memory
	  del raw_rest
	  del config_rest

	  #Creating "fake" events for the mem task data (to extract epochs)
	  events_mem = mne.make_fixed_length_events(raw_mem, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_mem = mne.pick_types(raw_mem.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_mem = mne.Epochs(raw_mem, events_mem, event_id = None, proj=False, picks=picks_mem, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs mem: ",len(epochs_mem.get_data()))
	  #Freeing memory
	  del raw_mem
	  del config_mem

	  #Creating "fake" events for the math task data (to extract epochs)
	  events_math = mne.make_fixed_length_events(raw_math, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_math = mne.pick_types(raw_math.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_math = mne.Epochs(raw_math, events_math, event_id = None, proj=False, picks=picks_math, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs math: ",len(epochs_math.get_data()))
	  #Freeing memory
	  del raw_math
	  del config_math

	  #Creating "fake" events for the motor task data (to extract epochs)
	  events_motor = mne.make_fixed_length_events(raw_motor, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_motor = mne.pick_types(raw_motor.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_motor = mne.Epochs(raw_motor, events_motor, event_id = None, proj=False, picks=picks_motor, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs motor: ",len(epochs_motor.get_data()))
	  #Freeing memory
	  del raw_motor
	  del config_motor

	  ####################### CREATING THE DATASET #######################

	  ##Extracting the x and y
	  x_rest = epochs_rest.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_rest = [0]*len(epochs_rest.get_data()) #All the epochs of task belong to the class 0
	  del epochs_rest
	  x_mem = epochs_mem.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_mem = [1]*len(epochs_mem.get_data()) #All the epochs of task belong to the class 0
	  del epochs_mem
	  x_math = epochs_math.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_math = [2]*len(epochs_math.get_data()) #All the epochs of task belong to the class 0
	  del epochs_math
	  x_motor = epochs_motor.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_motor = [3]*len(epochs_motor.get_data()) #All the epochs of task belong to the class 0
	  del epochs_motor

	  #Joining
	  if first_time:
	    X_train = np.concatenate((x_rest,x_mem,x_math,x_motor))
	    Y_train = np.concatenate((y_rest,y_mem,y_math,y_motor))
	    first_time = False
	  else:
	    X_train2 = np.concatenate((x_rest,x_mem,x_math,x_motor))
	    Y_train2 = np.concatenate((y_rest,y_mem,y_math,y_motor))
	    X_train = np.concatenate((X_train, X_train2))
	    Y_train = np.concatenate((Y_train, Y_train2))

	  del x_rest
	  del y_rest
	  del x_mem
	  del y_mem
	  del x_math
	  del y_math
	  del x_motor
	  del y_motor

	  #Normalizing data
	  #X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

	  print("\nAverage of the data: ", np.average(X_train))

	  #Freeing memory
	  del events_rest
	  del picks_rest
	  del events_mem
	  del picks_mem
	  del events_math
	  del picks_math
	  del events_motor
	  del picks_motor
	return X_train, Y_train, picks





# Reads the data from the given subjects and creates
# a dataset with that data (return X_train & Y_train)
# and cut the beginning of hte recording
def create_dataset_cut(list_subjects):
	#To concatenate the X & Y between patients
	first_time = True
	#Iterating among subjects
	for subject in list_subjects:
	  print("\n\n>>READING DATA FROM SUBJECT",subject, "...\n\n")
	  #Paths to the files
	  file_raw_rest =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_rest') #From real data
	  config_rest =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_rest') #From real data
	  file_raw_mem =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_mem') #From real data
	  config_mem =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_mem') #From real data 
	  file_raw_math =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_math') #From real data
	  config_math =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_math') #From real data
	  file_raw_motor =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_motor') #From real data
	  config_motor =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_motor') #From real data
	  #Reading resting data
	  raw_rest = mne.io.read_raw_bti(file_raw_rest, config_rest, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading memory data
	  raw_mem = mne.io.read_raw_bti(file_raw_mem, config_mem, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading math data
	  raw_math = mne.io.read_raw_bti(file_raw_math, config_math, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading motor data
	  raw_motor = mne.io.read_raw_bti(file_raw_motor, config_motor, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)

	  ####################### EXTRACTING EVENTS AND EPOCHS (second way -> Same size in the "chunks")#######################

	  #Creating "fake" events for the rest data (to extract epochs)
	  events_rest = mne.make_fixed_length_events(raw_rest, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_rest = mne.pick_types(raw_rest.info, meg=True, eeg=False, stim=False, eog=False)
	  picks = picks_rest
	  #Generating epochs
	  epochs_rest = mne.Epochs(raw_rest, events_rest, event_id = None, proj=False, picks=picks_rest, baseline=None, preload=True, verbose=False)
	  epochs_rest = epochs_rest[int(0.1*len(epochs_rest)):]
	  print("\n\nNumber of epochs rest: ",len(epochs_rest.get_data()))
	  #Freeing memory
	  del raw_rest
	  del config_rest

	  #Creating "fake" events for the mem task data (to extract epochs)
	  events_mem = mne.make_fixed_length_events(raw_mem, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_mem = mne.pick_types(raw_mem.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_mem = mne.Epochs(raw_mem, events_mem, event_id = None, proj=False, picks=picks_mem, baseline=None, preload=True, verbose=False)
	  epochs_mem = epochs_mem[int(0.1*len(epochs_mem)):]
	  print("\n\nNumber of epochs mem: ",len(epochs_mem.get_data()))
	  #Freeing memory
	  del raw_mem
	  del config_mem

	  #Creating "fake" events for the math task data (to extract epochs)
	  events_math = mne.make_fixed_length_events(raw_math, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_math = mne.pick_types(raw_math.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_math = mne.Epochs(raw_math, events_math, event_id = None, proj=False, picks=picks_math, baseline=None, preload=True, verbose=False)
	  epochs_math = epochs_math[int(0.1*len(epochs_math)):]
	  print("\n\nNumber of epochs math: ",len(epochs_math.get_data()))
	  #Freeing memory
	  del raw_math
	  del config_math

	  #Creating "fake" events for the motor task data (to extract epochs)
	  events_motor = mne.make_fixed_length_events(raw_motor, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_motor = mne.pick_types(raw_motor.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_motor = mne.Epochs(raw_motor, events_motor, event_id = None, proj=False, picks=picks_motor, baseline=None, preload=True, verbose=False)
	  epochs_motor = epochs_motor[int(0.1*len(epochs_motor)):]
	  print("\n\nNumber of epochs motor: ",len(epochs_motor.get_data()))
	  #Freeing memory
	  del raw_motor
	  del config_motor

	  ####################### CREATING THE DATASET #######################

	  ##Extracting the x and y
	  x_rest = epochs_rest.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_rest = [0]*len(epochs_rest.get_data()) #All the epochs of task belong to the class 0
	  del epochs_rest
	  x_mem = epochs_mem.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_mem = [1]*len(epochs_mem.get_data()) #All the epochs of task belong to the class 0
	  del epochs_mem
	  x_math = epochs_math.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_math = [2]*len(epochs_math.get_data()) #All the epochs of task belong to the class 0
	  del epochs_math
	  x_motor = epochs_motor.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	  y_motor = [3]*len(epochs_motor.get_data()) #All the epochs of task belong to the class 0
	  del epochs_motor

	  #Joining
	  if first_time:
	    X_train = np.concatenate((x_rest,x_mem,x_math,x_motor))
	    Y_train = np.concatenate((y_rest,y_mem,y_math,y_motor))
	    first_time = False
	  else:
	    X_train2 = np.concatenate((x_rest,x_mem,x_math,x_motor))
	    Y_train2 = np.concatenate((y_rest,y_mem,y_math,y_motor))
	    X_train = np.concatenate((X_train, X_train2))
	    Y_train = np.concatenate((Y_train, Y_train2))

	  del x_rest
	  del y_rest
	  del x_mem
	  del y_mem
	  del x_math
	  del y_math
	  del x_motor
	  del y_motor

	  #Normalizing data
	  #X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())

	  print("\nAverage of the data: ", np.average(X_train))

	  #Freeing memory
	  del events_rest
	  del picks_rest
	  del events_mem
	  del picks_mem
	  del events_math
	  del picks_math
	  del events_motor
	  del picks_motor
	return X_train, Y_train, picks


# Reads the data from the given subjects and creates
# a (split tasks) dataset with that data 
# (return x_rest, y_rest, x_mem, y_mem, x_math, y_math, x_motor & y_motor)
def create_dataset_separated_tasks(list_subjects):
	#To concatenate the X & Y between patients
	first_time = True
	#Iterating among subjects
	for subject in list_subjects:
	  print("\n\n>>READING DATA FROM SUBJECT",subject, "...\n\n")
	  #Paths to the files
	  file_raw_rest =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_rest') #From real data
	  config_rest =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_rest') #From real data
	  file_raw_mem =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_mem') #From real data
	  config_mem =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_mem') #From real data 
	  file_raw_math =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_math') #From real data
	  config_math =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_math') #From real data
	  file_raw_motor =  os.path.join(sys.path[0], 'subjects/c,rfDC_'+subject+'_motor') #From real data
	  config_motor =  os.path.join(sys.path[0], 'subjects/config_'+subject+'_motor') #From real data
	  #Reading resting data
	  raw_rest = mne.io.read_raw_bti(file_raw_rest, config_rest, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading memory data
	  raw_mem = mne.io.read_raw_bti(file_raw_mem, config_mem, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading math data
	  raw_math = mne.io.read_raw_bti(file_raw_math, config_math, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)
	  #Reading motor data
	  raw_motor = mne.io.read_raw_bti(file_raw_motor, config_motor, convert=False, head_shape_fname=None,
	          sort_by_ch_name=False, rename_channels=False, preload=False,
	          verbose=True)

	  ####################### EXTRACTING EVENTS AND EPOCHS (second way -> Same size in the "chunks")#######################

	  #Creating "fake" events for the rest data (to extract epochs)
	  events_rest = mne.make_fixed_length_events(raw_rest, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_rest = mne.pick_types(raw_rest.info, meg=True, eeg=False, stim=False, eog=False)
	  picks = picks_rest
	  #Generating epochs
	  epochs_rest = mne.Epochs(raw_rest, events_rest, event_id = None, proj=False, picks=picks_rest, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs rest: ",len(epochs_rest.get_data()))
	  #Freeing memory
	  del raw_rest
	  del config_rest

	  #Creating "fake" events for the mem task data (to extract epochs)
	  events_mem = mne.make_fixed_length_events(raw_mem, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_mem = mne.pick_types(raw_mem.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_mem = mne.Epochs(raw_mem, events_mem, event_id = None, proj=False, picks=picks_mem, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs mem: ",len(epochs_mem.get_data()))
	  #Freeing memory
	  del raw_mem
	  del config_mem

	  #Creating "fake" events for the math task data (to extract epochs)
	  events_math = mne.make_fixed_length_events(raw_math, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_math = mne.pick_types(raw_math.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_math = mne.Epochs(raw_math, events_math, event_id = None, proj=False, picks=picks_math, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs math: ",len(epochs_math.get_data()))
	  #Freeing memory
	  del raw_math
	  del config_math

	  #Creating "fake" events for the motor task data (to extract epochs)
	  events_motor = mne.make_fixed_length_events(raw_motor, id=1, start=0, stop=None, duration=3.0, first_samp=True, overlap=1)
	  #Choosing the channels (Only MEG in this case)
	  picks_motor = mne.pick_types(raw_motor.info, meg=True, eeg=False, stim=False, eog=False)
	  #Generating epochs
	  epochs_motor = mne.Epochs(raw_motor, events_motor, event_id = None, proj=False, picks=picks_motor, baseline=None, preload=True, verbose=False)
	  print("\n\nNumber of epochs motor: ",len(epochs_motor.get_data()))
	  #Freeing memory
	  del raw_motor
	  del config_motor

	  ####################### CREATING THE DATASET #######################	  

	  #Joining
	  if first_time:
	    ##Extracting the x and y
	    x_rest = epochs_rest.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_rest = [0]*len(epochs_rest.get_data()) #All the epochs of task belong to the class 0
	    del epochs_rest
	    x_mem = epochs_mem.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_mem = [1]*len(epochs_mem.get_data()) #All the epochs of task belong to the class 0
	    del epochs_mem
	    x_math = epochs_math.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_math = [2]*len(epochs_math.get_data()) #All the epochs of task belong to the class 0
	    del epochs_math
	    x_motor = epochs_motor.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_motor = [3]*len(epochs_motor.get_data()) #All the epochs of task belong to the class 0
	    del epochs_motor
	    first_time = False
	  else:
	    ##Extracting the x and y
	    x_rest2 = epochs_rest.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_rest2 = [0]*len(epochs_rest.get_data()) #All the epochs of task belong to the class 0
	    del epochs_rest
	    x_mem2 = epochs_mem.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_mem2 = [1]*len(epochs_mem.get_data()) #All the epochs of task belong to the class 0
	    del epochs_mem
	    x_math2 = epochs_math.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_math2 = [2]*len(epochs_math.get_data()) #All the epochs of task belong to the class 0
	    del epochs_math
	    x_motor2 = epochs_motor.get_data()*100000 #Scale by 1000 due to scaling sensitivity in deep learning
	    y_motor2 = [3]*len(epochs_motor.get_data()) #All the epochs of task belong to the class 0
	    del epochs_motor
	    x_rest = np.concatenate((x_rest, x_rest2))
	    y_rest = np.concatenate((y_rest, y_rest2))
	    x_mem = np.concatenate((x_mem, x_mem2))
	    y_mem = np.concatenate((y_mem, y_mem2))
	    x_math = np.concatenate((x_math, x_math2))
	    y_math = np.concatenate((y_math, y_math2))
	    x_motor = np.concatenate((x_motor, x_motor2))
	    y_motor = np.concatenate((y_motor, y_motor2))
	    del x_rest2
	    del y_rest2
	    del x_mem2
	    del y_mem2
	    del x_math2
	    del y_math2
	    del x_motor2
	    del y_motor2


	  #Normalizing data
	  #X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())


	  #Freeing memory
	  del events_rest
	  del picks_rest
	  del events_mem
	  del picks_mem
	  del events_math
	  del picks_math
	  del events_motor
	  del picks_motor
	return x_rest, y_rest, x_mem, y_mem, x_math, y_math, x_motor, y_motor, picks



#Writes a file with a summary of the experiment and the result of it
def gen_exp_summary(result, subjects, net_properties):
    experiments_folders_list = os.listdir(path='Experiments/EEGNet')
    temp_numbers=[]
    for folder in experiments_folders_list:
        number = re.findall(r'\d+', folder)
        if(len(number)>0):
            temp_numbers.append(int(number[0]))
    num = max(temp_numbers)
    #Writing the file
    filename = "Experiments/EEGNet/Experiment"+str(num)+"/Experiment_Summary"+str(num)+".txt"
    with open(filename, "w") as file:
    	file.write(">>Trained over the next list of subjects: \n  - "+str(subjects[0]))
    	file.write("\n\n>>Using the next stage of subjects: \n  - "+str(subjects[1]))
    	file.write("\n\n>>Test over the next subjects: \n  - "+str(subjects[2])+"\n  - Result -> loss: "+str(result[0])+" - acc: "+str(result[1]))
    	file.write("\n\n>>Properties EEGNet: \n  - "+str(net_properties[0])+" kernel lenght \n  - "+str(net_properties[1])+" temporal filters\n  - "+str(net_properties[2])+" dropout\n  - "+str(net_properties[3])+" pointwise filters")


