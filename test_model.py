import os 
import numpy as np
import random 
import argparse
from tensorflow.keras.models import load_model
from utils import *


ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True,
				help="path to output model")
ap.add_argument("-i", "--input", required=True,
				help="path to input image")
args = vars(ap.parse_args())


print("[INFO] Load the model.....")
model = load_model(args['model'])
print("[INFO] model loaded......")


print("\n[INFO] Getting the Image to be predicted......")
images, labels = preprocessing(args['input'])

print("\n[INFO] Prediction time...... ")
n = random.randint(1,100)
predictions(images[n], model)