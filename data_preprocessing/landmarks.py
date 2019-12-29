import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

global basedir, image_paths, target_size
basedir = './Datasets/celeba'
images_dir = os.path.join(basedir, 'img')
labels_filename = 'labels.csv'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data_preprocessing/shape_predictor_68_face_landmarks.dat')
# The is a face detector
# Find frontal human faces in an image using 68 landmarks
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)  # shape.num_parts=68
    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


# Get the distance from the face to the bounding
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


#detect the landmarks of the face, and then return the image and the landmarks
def run_dlib_shape(image,detector,predictor):
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    num_faces = len(rects)
    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region
        temp_shape = predictor(gray, rect)
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
    return dlibout, resized_image


def extract_features_labels(task):
    # Extracts the landmarks features
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None

    labels_file = open(os.path.join(basedir, labels_filename), 'r')

    lines = labels_file.readlines()
    if task == 'A1':
        gender_labels = {line.split(',')[0]: int(line.split(',')[2]) for line in lines[1:]} #gender label
        print('Gender detection applied')
    if task == 'A2':
        gender_labels = {line.split(',')[0]: int(line.split(',')[3]) for line in lines[1:]} #emotion label
        print('Emotion detection applied')
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        # output the path of images
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('/')[-1].split('\\')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img,detector,predictor)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[file_name])
    # convert to array
    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1) / 2  # simply converts the -1 into 0
    return landmark_features, gender_labels

