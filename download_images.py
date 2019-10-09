'''
The dataset used is available on the Kaggle website by the 
name: CAPTCHA images.

Either you can download it from the site or run this script to download it 
in your directory

We used Kaggle Python API
'''


import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files("fournierp/captcha-version-2-images",
									path="Datasets/",
									unzip=True)




