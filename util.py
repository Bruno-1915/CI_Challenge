"""Utilities
"""
import re
import cv2
import base64

import numpy as np

from PIL import Image
from io import BytesIO

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

path_model = './models/model_hm.h5'

#use = 'MobileNet'
use = 'OwnModel'

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")



def load_model():
    if use == 'MobileNet':
	       return MobileNetV2(weights='imagenet')
    else:
        return keras.models.load_model(path_model)

def get_sized_image(img):
    if use == 'MobileNet':
        img = img.resize((224, 224))

        # Preprocessing the image
        x = image.img_to_array(img)
        if len(x.shape) > 2 and x.shape[2] == 4:
            #convert the image from RGBA2RGB
            x = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
            # print('Changing channel numbers')
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)

    else:
        img = img.resize((28, 28))
        img = cv2.cvtColor(image.img_to_array(img), cv2.COLOR_BGR2GRAY)
        x = img.reshape(1,28,28,1)

    return x

def model_predict(img, model):
    x = get_sized_image(img)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = process_input(x)
    preds = model.predict(x)
    return preds

def process_input(x):
    if use == 'MobileNet':
        return preprocess_input(x, mode='tf')
    else:
        return x

def decode_preds(prediction):
    if use == 'MobileNet':
        pred_class = decode_predictions(prediction, top=1)
        return str(pred_class[0][0][1])               # Convert to string
    else:
        dic = [str(i) for i in range(10)] + ['Holmusk Logo']
        return dic[np.argmax(prediction)]
