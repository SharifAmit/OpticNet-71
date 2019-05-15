import tensorflow as tf
import keras
import argparse
import keras.backend as K
from keras.models import load_model
import cv2
import numpy as np


def print_pred(preds,classes):

    preds = preds.ravel()
    y = len(classes)
    x=""
    for i in range(y):
        preds_rounded = np.around(preds,decimals=4)
        x = x+classes[i]+": "+str(preds_rounded[i])+"%"
        if i!=(y-1):
            x = x+", "
        
        else:
            None
    print(x)

def image_preprocessing(img):
    img = cv2.imread(img)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    img = 1.0*img/255

    return img

def inference(img,weights,dataset):

    if dataset=='Srinivasan2014':
        classes=['AMD', 'DME','NORMAL']
    else:
        classes = ['CNV', 'DME','DRUSEN','NORMAL']

    processsed_img = image_preprocessing(img)
    K.clear_session()
    model = load_model(weights)
    
    preds = model.predict(processsed_img,batch_size=None,steps=1)
  
   
    print_pred(preds,classes)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True, help='path/to/image')
    parser.add_argument('--weights', type=str, required=True, help='Weights for prediction')
    parser.add_argument('--dataset', type=str, required=True, help='Choosing between 2 OCT datasets', choices=['Srinivasan2014','Kermany2018'])
    args = parser.parse_args()
    inference(args.imgpath, args.weights, args.dataset)
