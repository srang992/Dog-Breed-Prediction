# Dog Breed Prediction

For a very long time I am searching for an good project which can give me an better understanding on Convolutional Neural Network. After a long research, I came upon with this idea. In this project I am taking a dataset consisting of 120 types of dog breed, I train a CNN model for Predicting those dog breed perfectly. In this project I first learned about Transfer Learning. 

Here is an Example:

<br>

![Screenshot (28)](https://user-images.githubusercontent.com/86141125/148406871-0d8dcd9a-7dc4-4d16-9c45-a0b6ce16a90a.png)

<br>

Now this is not enough for describing my project. Let's dive deeper.

Collecting Data
--------------------

This data is collected from Kaggle using Kaggle API. For accessing Kaggle API, we need to go first in our profile in Kaggle and download a .json file containing our username and an api key. this file is necessary for accessing the Kaggle API. After that, we will install Kaggle API using pip installation and setting up kaggle using Kaggle API.
```
# install the Kaggle API client.
!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle, so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json
```

After that setup, we make a directory where we store our data after downloading from kaggle. Here I named that directory as dog_dataset.

```
# Creating directory and changing the current working directory
!mkdir dog_dataset
%cd dog_dataset
```

Now let's search for our dataset. Searching Kaggle for the required dataset using search option(-s) with title 'dogbreedidfromcomp'. We can also use different search options like searching competitions, notebooks, kernels, datasets, etc.

```
# Searching for dataset
!kaggle datasets list -s dogbreedidfromcomp
```
After running this, our output will look something like this ---

![Screenshot (34)](https://user-images.githubusercontent.com/86141125/148728537-b887e8ff-94ba-41e7-a6b6-b3c6be52aa4b.png)

It seems that we found our required data. Now we download that data and unzip it. we are also remove the irrelevant files.

```
# Downloading dataset and coming out of directory
!kaggle datasets download catherinehorng/dogbreedidfromcomp
%cd ..

# Unzipping downloaded file and removing unusable file
!unzip dog_dataset/dogbreedidfromcomp.zip -d dog_dataset
!rm dog_dataset/dogbreedidfromcomp.zip
!rm dog_dataset/sample_submission.csv
```

That's for our data collections. Now let's jump into data preprocessing.

Data Preprocessing
---------------------------

At first we import some of the required library.

```
# Important library imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
```

Next, we load the labels data into dataframe and viewing it. Here we analysed that labels contains 10222 rows and 2 columns.

![Screenshot (35)](https://user-images.githubusercontent.com/86141125/148729960-7f8784ff-5049-43ff-94fe-a942d507a1e3.png)

As we are working with the classification dataset first we need to one hot encode the target value i.e. the classes. After that we will read images and convert them into numpy array and finally normalizing the array.

```
# this will sort the dog breed list and display the total number of unique breeds
classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)

print("Total Unique Breed: ", n_classes)


# one-hot encoding the breeds
class_to_num = dict(zip(classes, range(n_classes)))


# function for converting images into numpy array
input_size = (331,331,3)

def images_to_array(directory, label_dataframe, target_size = input_size):

  image_labels = label_dataframe['breed']
  images = np.zeros((len(label_dataframe), 331, 331, 3), dtype = np.uint8)
  y = np.zeros((len(label_dataframe), 1), dtype = np.uint8)

  for index, image_name in enumerate(tqdm(label_dataframe['id'].values)):

    img_dir = os.path.join(directory, image_name + '.jpg')

    img = image.load_img(img_dir, target_size = target_size)

    images[index] = img

    dog_breed = image_labels[index]

    y[index] = class_to_num[dog_breed]

  y = to_categorical(y)

  return images,y
  ```
