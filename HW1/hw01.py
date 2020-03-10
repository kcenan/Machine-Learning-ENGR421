
#import libs
import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix

#read files
imagesdf = pd.read_csv('hw01_images.csv',header=None)
labeldf = pd.read_csv('hw01_labels.csv',header=None)

#split data into train and test
train_x = imagesdf.iloc[0:200]
test_x = imagesdf.iloc[-200:]
train_y = labeldf.iloc[0:200]
test_y = labeldf.iloc[-200:]

#get male and female set from training data
female_df = train_x.iloc[train_y[train_y[0]==1].index,:]
male_df = train_x.iloc[train_y[train_y[0]==2].index,:]

#get prior probabilities
priors = (len(train_y[train_y[0]==1]) / len(train_y),len(train_y[train_y[0]==2]) / len(train_y))

#get means and deviations
means = [female_df.apply(lambda x : np.array(x).mean()), male_df.apply(lambda x : np.array(x).mean())]
deviations = [female_df.apply(lambda x : np.array(x).std()), male_df.apply(lambda x : np.array(x).std())]

#score func
def score_func(x,mu,sigma,prob):
    result = sum((-1/2*np.log(2*math.pi)) - (np.log(sigma)) - ((x-mu)*(x-mu)/(2*sigma*sigma))) + np.log(prob)
    return result

#decider func for class selection
def decide_result(x):
    if score_func(x,means[0],deviations[0],0.1) > score_func(x,means[1],deviations[1],0.9):
        return 1
    else:
        return 2

#result on train data and confussion matrix
train_y_hat = train_x.apply(lambda x: decide_result(x), axis=1)
print(confusion_matrix(train_y, train_y_hat))

#result on test data and confussion matrix
test_y_hat = test_x.apply(lambda x: decide_result(x), axis=1)
print(confusion_matrix(test_y, test_y_hat))