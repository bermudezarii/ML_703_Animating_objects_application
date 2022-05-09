"""
    This file creates a split of data between training and testing (cross validation)
    and save the path of the images in CSV files.
"""

import glob
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from natsort import natsorted

# Reads all files in the positive and negative directories
positive_images = glob.glob('../dataset/Positive/*')
negative_images = glob.glob('../dataset/Negative/*')

# Balance the dataset by downsampling the majority class
min_class = positive_images if len(positive_images) < len(negative_images) else negative_images
positive_images = random.sample(positive_images, len(min_class))
negative_images = random.sample(negative_images, len(min_class))

# Merge positive and negative images
positive_images = natsorted(positive_images)
negative_images = natsorted(negative_images)
images = positive_images + negative_images
labels = [1]*len(positive_images) + [0]*len(negative_images)

# Shuffle the final dataset
# images, labels = shuffle(images_, labels_)

# Save the whole dataset
df = pd.DataFrame({'ID_IMG':images, 'LABEL': labels}, columns=['ID_IMG', 'LABEL'])
df.to_csv('../dataset/whole_dataset.csv')

# Creates a training (80%) and testing (20%) split and save the files
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=False)

df_train = pd.DataFrame({'ID_IMG':X_train, 'LABEL': y_train}, columns=['ID_IMG', 'LABEL'])
df_train.to_csv('../dataset/train_data.csv')

df_test = pd.DataFrame({'ID_IMG':X_test, 'LABEL': y_test}, columns=['ID_IMG', 'LABEL'])
df_test.to_csv('../dataset/test_data.csv')
