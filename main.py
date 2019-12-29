import A1.gender_model as model_A1
import A2.emotion_model as model_A2
import B1.face_model as model_B1
import B2.eye_model as model_B2
import data_preprocessing.data_preprocessing_A as dpa
import data_preprocessing.data_preprocessing_B as dpb
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ======================================================================================================================
# Task A1
X_train_A1, X_test_A1, y_train_A1, y_test_A1 = dpa.get_data_A1()
clf, acc_A1_train, acc_A1_val = model_A1.train(X_train_A1, y_train_A1) # Train model based on the training set
acc_A1_test = model_A1.test(clf, X_test_A1,y_test_A1)   # Test model based on the test set.
print('TA1:{},{},{};'.format(acc_A1_train, acc_A1_val, acc_A1_test))



# ======================================================================================================================
# Task A2
X_train_A2, X_test_A2, y_train_A2, y_test_A2 = dpa.get_data_A2()
clf, acc_A2_train, acc_A2_val = model_A2.train(X_train_A2, y_train_A2)
acc_A2_test = model_A2.test(clf, X_test_A2,y_test_A2)
print('TA2:{},{},{};'.format(acc_A2_train, acc_A2_val, acc_A2_test))


# ======================================================================================================================
# Task B1
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
path = "./Datasets/cartoon_set/"
data_train, data_val, data_test = dpb.get_data_B1(path, mark='face_shape')
model_B1 = model_B1.Face_Model()
acc_B1_train, acc_B1_val = model_B1.train(data_train, data_val)
acc_B1_test = model_B1.test(data_test)
print('TB1:{},{},{};'.format(acc_B1_train, acc_B1_val, acc_B1_test))


# ======================================================================================================================
# Task B2
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
path = "./Datasets/cartoon_set/"
data_train, data_val, data_test = dpb.get_data_B2(path, mark='eye_color')
model_B2 = model_B2.Eye_Model()
acc_B2_train, acc_B2_val = model_B2.train(data_train, data_val)
acc_B2_test = model_B2.test(data_test)
print('TB2:{},{},{};'.format(acc_B2_train, acc_B2_val, acc_B2_test))
