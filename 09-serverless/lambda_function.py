import os
import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
from keras_image_helper import create_preprocessor


onnx_model_path = os.getenv("MODEL_NAME","hair_classifier_v1.onnx")
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"]) #providers=["CPUExecutionProvider"]

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name
#print(input_name)
#print(output_name)


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

#image_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
target_size = 200
#image = download_image(image_url)
#image = prepare_image(image, (target_size, target_size))


def preprocess_pytorch_style(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)  

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)

preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(target_size, target_size))

def predict(url):
    image = download_image(url)
    image = prepare_image(image, (target_size, target_size))
    X = preprocessor.convert_to_tensor(image)
    result = session.run([output_name], {input_name: X})
    predictions = result[0][0].tolist()
    return predictions

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

#answer = lambda_handler(event, None)
#print(answer)