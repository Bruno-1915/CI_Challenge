import pytest
import warnings
from PIL import Image
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from CI_Challenge.util import *

def test_basic():
    assert str(4) == '4'

def test_model_can_predict():
   model = load_model()
   assert hasattr(model, 'predict')

def test_input_shape_expected_shape():
    model = load_model()
    img = Image.open('./tests/0.png')
    x = get_sized_image(img)
    model_iput_shape = model.get_config()['layers'][0]['config']['batch_input_shape'][1:]
    assert x.shape[1:] == model_iput_shape

def test_predict():
    model = load_model()
    image = Image.open('./tests/dog.jpg')
    model_predict(image, model)
