import numpy as np 
import string
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
symbols = string.ascii_lowercase + '0123456789'

def preprocessing(path):

	print("[INFO] Processing Images.......")
	n_samples= len(os.listdir(path))
	

	# variables for data and labels 
	X = np.zeros((n_samples , 50 , 200 ,1 ))  # (samples , height , width , channel)
	y = np.zeros((n_samples,5, 36 ))       #(samples , captcha characters , ascii char + numbers)

	for i , image in enumerate(os.listdir(path)):
		img = cv2.imread(os.path.join(path, image) , cv2.IMREAD_GRAYSCALE)
		targets = image.split('.')[0]
		if len(targets)<6:

			img = img/255.0
			img = np.reshape(img , (50,200,1))
			#find the char and one hot encode it to the target
			targ = np.zeros((5,36))

			for l , char in enumerate(targets):

				idx = symbols.find(char)
				targ[l , idx] = 1

			X[i] = img
			y[i,: ,:] = targ

	print("[INFO] Processing Finishes.....")
	return X,y




def predictions(image ,model):

	image_in = np.expand_dims(image , axis=0)

	print("[INFO] predicting the CAPTCHA")
	result = model.predict(image_in)
	result = np.reshape(result ,(5,36))
	indexes =[]
	for i in result:
	    indexes.append(np.argmax(i))
	    
	label=''
	for i in indexes:
	    label += symbols[i]

	placard = np.zeros((100,300))
	cv2.putText(placard, label, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255)) 
	cv2.imshow('output' , image.reshape((50,200)))
	cv2.imshow("playcard" , placard)
	#plt.title(label)
	print(label)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


