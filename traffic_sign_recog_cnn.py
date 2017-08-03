import numpy as np
from skimage import io, color, exposure, transform
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import os
import glob
import h5py
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator #for data augmentation.... not done here but can be added
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])



#use h5py instead of pickle


try:
    with  h5py.File('traff.h5') as hf: 
        X, Y = hf['imgs'][:], hf['labels'][:]
    print("Loaded images from traff.h5")
    
except (IOError,OSError, KeyError):  
    print("Error in reading traff.h5. Processing all images...")
    root_dir = 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
	    #img = preprocess_img(imread(img_path))
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processing done {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('traff.h5','w') as hf:
        hf.create_dataset('imgs', data=X)
        hf.create_dataset('labels', data=Y)


#keras
def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, IMG_SIZE, IMG_SIZE), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

model = cnn_model()
#SGD 
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

#training.... finally 
batch_size = 32
nb_epoch = 30

model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2, shuffle=True, callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5',save_best_only=True)]
            )


#loading data

test = pd.read_csv('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/GT-final_test.csv',sep=';')

X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/',file_name)
    X_test.append(preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    
X_test = np.array(X_test)
y_test = np.array(y_test)

#warn("The default mode, 'constant', will be changed to 'reflect' in ") lite le

y_pred = model.predict_classes(X_test)
s_acc = 0.0
for i in range(len(y_pred)):
    if y_test[i] == y_pred[i]:
        s_acc = s_acc+1
s_acc = s_acc*1.0
print("Test accuracy = ",s_acc/np.size(y_pred))
s_acc = 0.0
	
cm = metrics.confusion_matrix(y_test, y_pred)
print (cm)
np.savetxt("cm.csv", cm, delimiter=",")

model.summary()

#adversarial examples are not recognized correctly


#1st try ------->0.893198556784
#2nd try ------->0.928934215607
#3rd try ------->0.957601791134
#Test accuracy = 0.979930324621.....4th try
#Epoch 30/30  <--- Last epoch
#31367/31367 [==============================] - 1590s - loss: 0.0060 - acc: 0.9981 - val_loss: 0.0109 - val_acc: 0.9967

