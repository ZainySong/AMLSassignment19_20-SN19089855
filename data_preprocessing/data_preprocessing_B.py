import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import color
from keras.utils import np_utils
import cv2
import numpy as np

def get_data_B1(path, mark='face_shape'):
    image_size = 50
    label_path = path + "labels.xlsx"
    label = pd.read_excel(label_path)
    label = label[['file_name', mark]]
    x_feat = []
    y = label[mark].tolist()
    y = np_utils.to_categorical(y)
    for name in label['file_name'].tolist():
        pic_path = path + "img/" + name
        #print(pic_path)
        img = cv2.imread(pic_path)
        # Ignore the color information of images in this task
        img = color.rgb2gray(img)
        img = cv2.resize(img, (image_size, image_size))
        img = img.reshape(image_size, image_size, 1)
        x_feat.append(img)
    x_feat = np.array(x_feat)
    x_feat = np.vstack(x_feat).reshape(-1, image_size, image_size, 1)

    x_feat = x_feat / 255
    x_train, x_other, y_train, y_other = train_test_split(x_feat, y, test_size=0.4, random_state=2019)
    x_val, x_test, y_val, y_test = train_test_split(x_other, y_other, test_size=0.5, random_state=2019)
    return [x_train, y_train], [x_val, y_val], [x_test, y_test]


def get_data_B2(path,mark='face_shape'):
    image_size = 150
    label_path = path + "labels.xlsx"
    label = pd.read_excel(label_path)
    label = label[['file_name',mark]]
    x_feat = []
    y = label[mark].tolist()
    y = np_utils.to_categorical(y)
    for name in label['file_name'].tolist():
        pic_path = path + "img/" + name
        #print(pic_path)
        img = cv2.imread(pic_path)
        img = cv2.resize(img,(image_size, image_size))
        x_feat.append(img)
    x_feat = np.array(x_feat)
    x_feat = x_feat / 255
    x_train,x_other,y_train,y_other = train_test_split(x_feat,y,test_size=0.4,random_state=2019)
    x_val,x_test,y_val,y_test = train_test_split(x_other,y_other,test_size=0.5,random_state=2019)
    return [x_train,y_train],[x_val,y_val],[x_test,y_test]
