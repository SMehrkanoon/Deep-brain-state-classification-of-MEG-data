import argparse
import tensorflow
import data_utils as utils
from Cascade import Cascade
from MultiviewAttention import MultiviewAttention
from MultiviewSelfGlobalAttention import MultiviewSelfGlobalAttention
import time
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--attention', type=str, default="self", help = "Please choose which type of model to load : no attention, \
                    self-attention only (self), or self + global attention (global), by default self attention", choices=["no","self","global"])
args = parser.parse_args()

if args.attention == "no":
    model_dir = "best_models/best_no_attention"
    depth = 100
elif args.attention == "self":
    model_dir = "best_models/best_self_attention"
    depth = 10
elif args.attention == "global":
    model_dir = "best_models/best_self_global_attention" 
    depth = 10


model = tensorflow.keras.models.load_model(model_dir)
list_subjects_test = ['204521','212318','162935','601127','725751','735148']
accuracies_temp = []

#Creating dataset for testing
for subject in list_subjects_test:
    start_testing = time.time()
    print("\nTesting on subject", subject)
    subject_files_test = []
    for item in utils.test_files_dirs:
        if subject in item:
            subject_files_test.append(item)
              
    number_workers_testing = 10
    number_files_per_worker = len(subject_files_test)//number_workers_testing
    if args.attention == "no" or args.attention == "global":
        X_test, Y_test = utils.multi_processing_cascade(subject_files_test,number_files_per_worker,number_workers_testing,depth)
    else:
        X_test, Y_test = utils.multi_processing_multiview(subject_files_test,number_files_per_worker,number_workers_testing,depth)
    result = model.evaluate(X_test, Y_test, batch_size = 64,verbose=1)
    
    accuracies_temp.append(result[1])
    print("Timespan of testing is : {}".format(time.time() - start_testing))
avg = sum(accuracies_temp)/len(accuracies_temp)
print("\n\nAverage testing accuracy : {0:.2f}".format(avg))
print("Standard deviation: {}".format(np.std(accuracies_temp)))
