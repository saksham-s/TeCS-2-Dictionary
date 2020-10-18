from __future__ import print_function
from sklearn import decomposition
from scipy.misc import imread
import numpy as np
import glob
import sys
import os
import cPickle as pickle
from scipy.misc import imread,imresize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset="plastic_surger"

a=np.load("plastic_surgery_numpy.npz")

x_train, x_test, y_train, y_test = train_test_split(a['x'],a['y'], test_size=0.2, random_state=42)

with open('dictionary_color_model.pkl','rb') as fin:
	dictionary_learner=pickle.load(fin)
transform_train=dictionary_learner.transform(x_train)
np.savez(dataset+"_numpy_color_train", x=transform_train, y=y_train)
transform_test=dictionary_learner.transform(x_test)
np.savez(dataset+"_numpy_color_test", x=transform_test, y=y_test)

with open('dictionary_shapes_model.pkl','rb') as fin:
	dictionary_learner=pickle.load(fin)
transform_train=dictionary_learner.transform(x_train)
np.savez(dataset+"_numpy_shapes_train", x=transform_train, y=y_train)
transform_test=dictionary_learner.transform(x_test)
np.savez(dataset+"_numpy_shapes_test", x=transform_test, y=y_test)

with open('dictionary_texture_model.pkl','rb') as fin:
	dictionary_learner=pickle.load(fin)
transform_train=dictionary_learner.transform(x_train)
np.savez(dataset+"_numpy_texture_train", x=transform_train, y=y_train)
transform_test=dictionary_learner.transform(x_test)
np.savez(dataset+"_numpy_texture_test", x=transform_test, y=y_test)

with open('dictionary_symmetry_model.pkl','rb') as fin:
	dictionary_learner=pickle.load(fin)
transform_train=dictionary_learner.transform(x_train)
np.savez(dataset+"_numpy_symmetry_train", x=transform_train, y=y_train)
transform_test=dictionary_learner.transform(x_test)
np.savez(dataset+"_numpy_symmetry_test", x=transform_test, y=y_test)

