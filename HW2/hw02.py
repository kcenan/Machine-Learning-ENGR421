#import libs
import pandas as pd
import numpy as np
import math

#read features and labels
imagesdf = pd.read_csv('hw02_images.csv',header=None)
labeldf = pd.read_csv('hw02_labels.csv',header=None)

#read weights for regression initial state
i_W_df = pd.read_csv('initial_W.csv',header=None)
i_w0_df = pd.read_csv('initial_w0.csv',header=None)

#split data into train and test data
train_x = imagesdf.iloc[0:500]
test_x = imagesdf.iloc[-500:]
train_y = labeldf.iloc[0:500]
test_y = labeldf.iloc[-500:]

#sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

