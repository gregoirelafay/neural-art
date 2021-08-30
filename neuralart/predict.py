import tensorflow as tf
from tensorflow.keras import models
class Predict():
    def __init__(self, image):
        self.image = image
        self.model_path

    def decode_image(self, img_height, img_width):
        img = tf.io.decode_image(self.image, channels=3)
        self.decode_image = tf.image.resize(self.image, [img_height, img_width])

    def preprocess_image(self, model_name):
        models.load_model(filepath)


    def load_model(model_path):
        pass

    def get_prediction(img):
        model = load_model(model_path)
        model.predict(tf.expand_dims(img, axis=0))
