import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D, MaxPool2D, BatchNormalization, LeakyReLU
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Face_Model():
    def __init__(self):
        self.image_size = 50
        self.model = self.getmodel()
        self.batch_size = 32
        self.epochs = 4

    def getmodel(self):
        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False,
                                input_shape=(self.image_size, self.image_size, 1)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())

        model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(5, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train, val):
        history = self.model.fit(train[0], train[1],
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=(val[0], val[1])
                                 )

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("face_shape.png")
        return history.history['acc'][-1], history.history['val_acc'][-1]

    def test(self, test):
        pre = self.model.predict_classes(test[0])
        real_y_test = np.argmax(test[1], axis=1)
        #print('pre', pre.shape)
        #print('test', real_y_test.shape)

        return accuracy_score(real_y_test, pre)