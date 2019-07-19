import os
import random
import cv2
import sys
import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.optimizers import SGD

IMAGE_SIZE = 224
# 训练图片大小
epochs = 50
# 遍历次数
batch_size = 32
# 批量大小
nb_train_samples = 256*2
# 训练样本总数
nb_validation_samples = 64*2
# 测试样本总数
 
# 样本图片所在路径
FILE_PATH = 'Gender.h5'
# 模型存放路径

def gender(img_path):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    Gender = Model()
    Gender.load()    
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    gender = check(Gender,img, faces)
    print(gender)
    return gender

def check(Gender, img, rects):
        text = "can not check"
        for x, y, w, h in rects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 00), 2)
            face = img
            face = cv2.resize(face,(224,224))
            if Gender.predict(face)==1:
                text = "Male"
            else:
                text = "Female"
        return text   
 
class Model(object):

    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid')) 

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model.load_weights(file_path)

    def predict(self, image):
        # 预测样本分类
        img = image.resize((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        img = image.astype('float32')
        img /= 255
        #归一化
        result = self.model.predict(img)
        print(result)
        # 概率
        result = self.model.predict_classes(img)
        print(result)
        # 0/1

        return result[0]

 
if __name__ == '__main__':     
    gender('14.png')
    