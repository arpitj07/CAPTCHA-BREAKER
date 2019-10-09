import os 
import numpy as np 
import string
import cv2
import os
import argparse
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dropout
from tensorflow.keras.layers import Flatten , Dense ,Layer , BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model , Input
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#defining argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-d' , "--dataset" , required=True,
				help='Path to dataset')

ap.add_argument("-m", "--model", required=True,
				help="path to output model")
args = vars(ap.parse_args())

symbols = len(string.ascii_lowercase + "0123456789")

def myModel():
    
    inputs = Input(shape=(50,200,1) , name='image')
    x= Conv2D(16, (3,3),padding='same',activation='relu')(inputs)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x = BatchNormalization()(x)
    out_flat= Flatten()(x)
    
    #char-1
    dense_1 = Dense(64 , activation='relu')(out_flat)
    dropout_1= Dropout(0.5)(dense_1)
    out_1 = Dense(symbols , activation='sigmoid' , name='char_1')(dropout_1)
    
    #char-2
    dense_2 = Dense(64 , activation='relu')(out_flat)
    dropout_2= Dropout(0.5)(dense_2)
    out_2 = Dense(symbols , activation='sigmoid' , name='char_2')(dropout_2)
    
    #char-3
    dense_3 = Dense(64 , activation='relu')(out_flat)
    dropout_3= Dropout(0.5)(dense_3)
    out_3 = Dense(symbols , activation='sigmoid' , name='char_3')(dropout_3)
    
    #char-4
    dense_4 = Dense(64 , activation='relu')(out_flat)
    dropout_4= Dropout(0.5)(dense_4)
    out_4 = Dense(symbols , activation='sigmoid' , name='char_4')(dropout_4)
    
    #char-5
    dense_5 = Dense(64 , activation='relu')(out_flat)
    dropout_5= Dropout(0.5)(dense_5)
    out_5 = Dense(symbols , activation='sigmoid' , name='char_5')(dropout_5)
    
    model_out = Model(inputs=inputs , outputs=[out_1 , out_2 , out_3 , out_4 , out_5])
    
    return model_out
    



#Getting Data and labels
X , y = preprocessing(args['dataset'])
trainX, testX , trainY , testY = train_test_split(X, y,
												test_size=0.2, 
												random_state=42)
#target values
labels = {'char_1': trainY[:,0,:], 
         'char_2': trainY[:,1,:],
         'char_3': trainY[:,2,:],
         'char_4': trainY[:,3,:],
         'char_5': trainY[:,4,:]}

test_labels = {'char_1': testY[:,0,:], 
         'char_2': testY[:,1,:],
         'char_3': testY[:,2,:],
         'char_4': testY[:,3,:],
         'char_5': testY[:,4,:]}
#initialize the model
print("[INFO] Compiling Model.....")
model = myModel()
opt = Adam(learning_rate=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

print("[INFO] training the model.....")
model.fit(trainX, labels, epochs=30, batch_size=128,validation_split=0.2)
print(model.summary())


print("[INFO] Evaluating network....")
results = model.evaluate(testX, test_labels , batch_size=32)
print(results)

#print(classification_report(np.argmax(testY) , np.argmax(results)))
print("[INFO] saving the model")
model.save(args['model'])