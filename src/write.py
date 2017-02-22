import numpy as np
from PIL import Image

import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def write_images(X, y):
    for label in np.unique(y):
        mkdir_p("data/"+str(label))

    for i in range(np.size(X)-1):
        image=X[i]
        label=y[i]
        im = Image.fromarray(image)
        im.save("data/"+str(label)+"/"+str(i)+".jpeg")

        if(i%1000==0):
            print("Wrote", i , "images...")