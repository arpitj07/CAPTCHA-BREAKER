# CAPTCHA-BREAKER

CAPTCHAs were designed to prevent computers from automatically filling out forms by verifying that you are a real person. But with the rise of deep learning and computer vision, they can now often be defeated easily. 


## Content:
-  [Dependencies](https://github.com/arpitj07/CAPTCHA-BREAKER/blob/master/README.md#dependencies-1)
-  [Dataset](https://github.com/arpitj07/CAPTCHA-BREAKER/blob/master/README.md#dataset-2)
-  [Prepocessing](https://github.com/arpitj07/CAPTCHA-BREAKER/blob/master/README.md#prepocessing-3) 
-  [Training](https://github.com/arpitj07/CAPTCHA-BREAKER/blob/master/README.md#training-4)
-  [Testing & Predictions]()
-  [Visaulisation]()


### Dependencies 
This project requires a lot of modules and packages. This can be installed from `requirement.txt` file using following command:

```
pip install -r requirements.txt for python 2.x
pip3 install -r requirements.txt for python 3.x

```

### Dataset

Data for the project is available on [Kaggle](https://www.kaggle.com/fournierp/captcha-version-2-images). We can download to our local system using pyton script `download_images.py`. Data will be downloaded to Dataset folder. Run the following command:

```
python download_images.py

```

### Prepocessing

The raw data need to be prepocessed before feeding into the model. All the code for the same is provided in `utils.py`


### Training 

After downloading data, prepocessing images and labels, we will train the model. We need to provide 2 arguments.
1) Path to Input images
2) Path to save the model
To train, run the command:

```
python train_model.py --dataset Dataset/ --model Output/

```

### Testing & Predictions:

Final step is to test your model. Pass the following arguments and run the code from `test_model.py`.
1) Input: Dataset to predict
2) Model: path to Saved model

```
python test_model.py  --input Datasets/ --model Output/saved_model.pb

```
