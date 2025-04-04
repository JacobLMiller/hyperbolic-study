from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
import numpy as np 
import json 

X,y = make_blobs(2 * 5000, 50,centers=10)

np.save("largeblobs.npy", X)
np.save("largeblobs_lab.npy", y)

# data = [{
#  "class": int(lab)   
# } for lab in y]

# with open("src/application/static/data/iris.json", 'w') as fdata:
#     json.dump(data,fdata)