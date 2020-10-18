from __future__ import print_function
from sklearn import decomposition
from scipy.misc import imread
import numpy as np
import glob
import sys
import os
import pickle


directory="./channel_input/*"

domain_data=np.zeros((len(glob.glob(directory)),12288))

i=0
image_name=[]
for imgcsv in glob.glob(directory):
	img = imread(imgcsv)
	print(imgcsv)
	flat=img.flatten()
	image_name.append(imgcsv)
	domain_data[i]=flat
	i+=1
	print(i)

dictionay_learner=decomposition.DictionaryLearning(n_components=4200, alpha=1,
 max_iter=500, tol=1e-08,
 fit_algorithm='lars', transform_algorithm='omp',n_jobs=1,
 verbose=True, split_sign=False, random_state=42)
dictionay_learner.fit(domain_data)

with open('dictionary_symmetry_model.pkl','wb') as fin:
	pickle.dump(dictionay_learner,fin)
