import tensorflow as tf
from tensorflow.keras import models

class Predict():
    def __init__(self, image=None, model=None):
        self.image = image
        self.class_names = ['abstract', 'color_field_painting', 'cubism', 'expressionism',
                            'impressionism', 'realism', 'renaissance', 'romanticism']
        self.model = model
        self.model_path = None
        self.decoded_image = None
        self.img_height = None
        self.img_width = None


    def decode_image(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

        img = tf.io.decode_image(tf.constant(self.image), channels=3)
        self.decoded_image = tf.image.resize(img,
                                             [self.img_height, self.img_width])

    def load_model(self,model_path):
        self.model_path = model_path
        self.model = models.load_model(model_path)

    def get_prediction(self):
        assert self.model, "Please load a model"
        return self.model.predict(tf.expand_dims(self.decoded_image, axis=0))
