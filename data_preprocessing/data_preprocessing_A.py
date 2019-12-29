import numpy as np
from sklearn.model_selection import train_test_split
import data_preprocessing.landmarks as lm

def get_data_A1():
    task = 'A1'
    X, y = lm.extract_features_labels(task)
    Y = np.array([y, -(y - 1)]).T
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    X_train = X_train.reshape(int(X_train.size/(68*2)), 68*2)
    y_train = list(zip(*y_train))[0]
    X_test = X_test.reshape(int(X_test.size/(68*2)), 68*2)
    y_test = list(zip(*y_test))[0]
    return X_train, X_test, y_train, y_test

def get_data_A2():
    task = 'A2'
    X, y = lm.extract_features_labels(task)
    Y = np.array([y, -(y - 1)]).T
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    X_train = X_train.reshape(int(X_train.size / (68*2)), 68*2)
    y_train = list(zip(*y_train))[0]
    X_test = X_test.reshape(int(X_test.size / (68*2)), 68*2)
    y_test = list(zip(*y_test))[0]
    return X_train, X_test, y_train, y_test


