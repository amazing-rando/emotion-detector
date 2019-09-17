from keras.utils import to_categorical
import numpy as np
from glob import glob
from random import shuffle
import cv2, pickle


#Declare image locations.
angry = glob("./faces/angry/*")
sad = glob("./faces/sad/*")
happy = glob("./faces/happy/*")
neutral = glob("./faces/neutral/*")

#Angry is the category that we have the fewest images for.
#Subsample other images to have a balanced data set.
maxlen = len(angry) - 1

#Shuffle images.
for emotion in [angry, sad, happy, neutral]:
    shuffle(emotion)
files = [angry, sad, happy, neutral]

#0,1,2,3 maps to: angry, sad, happy, neutral
labels = []
faces = []
for i, typ in enumerate(files):
    for img in typ[0:maxlen]:
        labels.append(i)
        faces.append(cv2.imread(img,0))

#Convert labels to one-hot encoded categorical variables.
labels = to_categorical(labels, 4)

#Save serialized data.
pklfile = open("data.pkl", 'ab') 
pickle.dump([faces, labels], pklfile)                      
pklfile.close() 
