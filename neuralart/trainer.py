import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models, applications
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, experiment_name='test_trainer'):
        self.experiment_name = experiment_name
        self.csv_filename_path = None
        self.image_folder_path = None
        self.model_folder_path = None
        self.model_filename = None
        self.batch_size = None
        self.buffer_size = None
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.class_names = ['abstract', 'color_field_painting', 'cubism', 'expressionism',
                            'impressionism', 'realism', 'renaissance', 'romanticism']
        self.num_classes = 8
        self.image_count = None
        self.img_height = None
        self.img_width = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model = None
        self.model_name = None
        self.trainable_layers = None
        self.random_rotation = None
        self.random_zoom = None
        self.epochs = None
        self.learning_rate = None
        self.history = None
        self.results = None

    def create_dataset_from_directory(self, image_folder_path, batch_size, img_height, img_width):
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.image_folder_path + 'train',
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=True)

        assert len(self.train_ds.class_names) == self.num_classes

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.image_folder_path + 'val',
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            label_mode='categorical',
            batch_size=self.batch_size)

        assert len(self.val_ds.class_names) == self.num_classes

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=self.image_folder_path + 'test',
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            label_mode='categorical',
            batch_size=self.batch_size)

        assert len(self.test_ds.class_names) == self.num_classes

        self.image_count = (int(
            len(list(self.train_ds)))+int(len(list(self.val_ds)))+int(len(list(self.test_ds))))*self.batch_size
        self.buffer_size = int(self.image_count/10)

        self.train_ds = self.conf_perf_train_ds_from_directory(self.train_ds)
        self.val_ds = self.conf_perf_val_test_ds_from_directory(self.val_ds)
        self.test_ds = self.conf_perf_val_test_ds_from_directory(self.test_ds)

    def create_dataset_from_csv(self, csv_filename_path, image_folder_path, batch_size, img_height, img_width):
        self.csv_filename_path = csv_filename_path
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        data = pd.read_csv(self.csv_filename_path)
        self.image_count = data.shape[0]
        self.buffer_size = int(self.image_count/10)
        assert set(list(data["movement"].unique())) == set(self.class_names)
        assert data["movement"].nunique() == self.num_classes

        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (list(self.image_folder_path + data.loc[data["split"] == "train", "file_name"]), data["movement"]))

        self.val_ds = tf.data.Dataset.from_tensor_slices(
            (list(self.image_folder_path + data.loc[data["split"] == "val", "file_name"]), data["movement"]))

        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (list(self.image_folder_path + data.loc[data["split"] == "test", "file_name"]), data["movement"]))

        self.train_ds = self.train_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.val_ds = self.val_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.test_ds = self.test_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)

        self.train_ds = self.conf_perf_train_ds_from_csv(self.train_ds)
        self.val_ds = self.conf_perf_val_test_ds_from_csv(self.val_ds)
        self.test_ds = self.conf_perf_val_test_ds_from_csv(self.test_ds)

    def get_label(self,label):
        class_loc = label == self.class_names
        class_indice = tf.argmax(class_loc) # Integer encode the label
        class_ohe = tf.one_hot(class_indice, self.num_classes)
        return class_ohe

    def decode_img(self,img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self,filepath, label):
        label = self.get_label(label)
        img = tf.io.read_file(filepath)
        img = self.decode_img(img)
        return img, label

    def conf_perf_train_ds_from_csv(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def conf_perf_val_test_ds_from_csv(self, ds):
        ds = ds.cache()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def configure_performance_train_ds_from_directory(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def configure_performance_val_test_ds_from_directory(self, ds):
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def build_model(self, model_name, trainable_layers=2, random_roration=0.3, random_zoom=0.3, learning_rate=0.001):
        self.model_name = model_name
        self.trainable_layers = trainable_layers
        self.random_rotation = random_roration
        self.random_zoom = random_zoom
        self.learning_rate=learning_rate

        assert self.model_name in {
            "VGG16", "ResNet50", "custom"}, "Choose a model among the following ones: 'VGG16', 'ResNet50', 'custom'"

        if self.model_name == "VGG16":
            layer_model = self.get_vgg16_model()

        if self.model_name == "ResNet50":
            layer_model = self.get_resnet50_model()

        if self.model_name in {"ResNet50", "VGG16"}:
            layer_model.trainable = False
            for layer in layer_model.layers[-self.trainable_layers:]:
                layer.trainable = True

        if self.model_name == 'custom':
            pass

        data_augmentation_layers = self.get_data_augmentation_layers()

        inputs = Input(shape=(self.img_height, self.img_width, 3))

        x = data_augmentation_layers(inputs)  # Are not applied to validation and test dataset (made inactive, tensorflow handle it)
        if self.model_name == "VGG16":
            x = applications.vgg16.preprocess_input(x)  # Does the rescaling
        if self.model_name == "ResNet50":
            x = applications.resnet50.preprocess_input(x)  # Does the rescaling
        x = layer_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax',
                            name='classification_layer')(x)

        self.model = Model(inputs, outputs)

        self.model.compile(optimizer=optimizers.Adamax(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_vgg16_model(self):
        return applications.VGG16(include_top=False,weights='imagenet',
                                  input_shape=(self.img_height,self.img_width,3),
                                  classes=self.num_classes)

    def get_resnet50_model(self):
        return applications.ResNet50(include_top=False, weights='imagenet',
                                  input_shape=(self.img_height,self.img_width,3),
                                  classes=self.num_classes)

    def get_data_augmentation_layers(self):
        return models.Sequential([
            RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3)),
            RandomRotation(self.random_rotation),
            RandomZoom(self.random_zoom)
            ])

    def run(self, epochs=100):
        self.epochs = epochs

        assert self.model, "Run the build_model() method first"
        assert self.train_ds and self.val_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"

        es = EarlyStopping(monitor='val_loss', patience=20,
                           mode='min', restore_best_weights=True)

        rlrp = ReduceLROnPlateau(
            monitor='val_loss', factor=0.4, patience=3, min_lr=1e-8)

        self.history = self.model.fit(
            self.train_ds,
            epochs=self.EPOCHS,
            validation_data=self.val_ds,
            callbacks=[es, rlrp],
            use_multiprocessing=True)

        return self.history

    def save_model(self, model_folder_path=None, model_filename=None):

        assert self.model, "Run the build_model() and run() methods first"

        recorded_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model_filename = model_filename
        self.model_folder_path = model_folder_path

        if not model_filename:
            self.model_filename = f"{recorded_time}-images_{self.image_count}-unfreeze_{self.trainable_layers}-batch_{self.batch_size}"

        if not model_folder_path:
            self.model_folder_path = os.path.join('..', 'models', self.experiment_name, self.model_name)

        self.model.save(os.path.join(self.model_folder_path, self.model_filename))

    def evaluate_model(self):
        assert self.test_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        assert self.model, "Run the build_model() method first"
        assert self.history, "Run the run() method first"

        self.results = self.model.evaluate(self.test_ds)

        return self.results

    def history_visualization(self):
        assert self.test_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        assert self.model, "Run the build_model() method first"
        assert self.history, "Run the run() method first"

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = self.history.epoch

        fig,ax = plt.subplots(1,2,figsize=(16, 10))
        ax[0].plot(epochs_range, acc, label='Training Accuracy')
        ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
        ax[0].legend(loc='lower right')
        ax[0].set_title('Training and Validation Accuracy')

        ax[1].plot(epochs_range, loss, label='Training Loss')
        ax[1].plot(epochs_range, val_loss, label='Validation Loss')
        ax[1].legend(loc='upper right')
        ax[1].set_title('Training and Validation Loss')
