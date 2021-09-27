import tensorflow as tf
from tensorflow.keras import Input, Model, layers, models, applications, optimizers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer():
    def __init__(self, experiment_name='test_trainer'):
        self.experiment_name = experiment_name
        self.csv_path = None
        self.image_folder_path = None
        self.save_model_path = None
        self.load_model_path = None
        self.batch_size = None
        self.buffer_size = None
        self.shuffle_dataframe = None
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.class_names = np.array(['abstract', 'color_field_painting', 'cubism', 'expressionism',
                            'impressionism', 'realism', 'renaissance', 'romanticism'])
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
        self.use_rlrp = None
        self.learning_rate = None
        self.history = None
        self.results = None

    def create_dataset_from_directory(self, image_folder_path, batch_size, img_height, img_width):
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=os.path.join(self.image_folder_path,'train'),
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical',
            shuffle=True)

        assert len(self.train_ds.class_names) == self.num_classes

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=os.path.join(self.image_folder_path,'val'),
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            label_mode='categorical',
            batch_size=self.batch_size)

        assert len(self.val_ds.class_names) == self.num_classes

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory=os.path.join(self.image_folder_path,'test'),
            labels='inferred',
            image_size=(self.img_height, self.img_width),
            label_mode='categorical',
            batch_size=self.batch_size)

        assert len(self.test_ds.class_names) == self.num_classes

        self.image_count = (int(
            len(list(self.train_ds)))+int(len(list(self.val_ds)))+int(len(list(self.test_ds))))*self.batch_size
        self.buffer_size = self.image_count

        self.train_ds = self.conf_perf_ds_from_directory(self.train_ds, train_split=True)
        self.val_ds = self.conf_perf_ds_from_directory(self.val_ds)
        self.test_ds = self.conf_perf_ds_from_directory(self.test_ds)

    def create_dataset_from_csv(self, csv_path, image_folder_path, batch_size, img_height, img_width, shuffle_dataframe=False):
        self.csv_path = csv_path
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.shuffle_dataframe = shuffle_dataframe

        data = pd.read_csv(self.csv_path)
        if self.shuffle_dataframe:
            data = data.sample(frac=1)
        self.image_count = data.shape[0]
        self.buffer_size = self.image_count
        assert set(list(data["style"].unique())) == set(self.class_names)
        assert data["style"].nunique() == self.num_classes

        self.train_ds = tf.data.Dataset.from_tensor_slices(([
            os.path.join(self.image_folder_path, i[1])
            for i in data.loc[data["split"] == "train","image_path"].iteritems()
        ], data.loc[data["split"] == "train", "style"]))

        self.val_ds = tf.data.Dataset.from_tensor_slices(([
            os.path.join(self.image_folder_path, i[1])
            for i in data.loc[data["split"] == "val", "image_path"].iteritems()
        ], data.loc[data["split"] == "val", "style"]))

        self.test_ds = tf.data.Dataset.from_tensor_slices(([
            os.path.join(self.image_folder_path, i[1])
            for i in data.loc[data["split"] == "test", "image_path"].iteritems()
        ], data.loc[data["split"] == "test", "style"]))

        self.train_ds = self.train_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.val_ds = self.val_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)
        self.test_ds = self.test_ds.map(
            self.process_path, num_parallel_calls=self.AUTOTUNE)

        self.train_ds = self.conf_perf_ds_from_csv(self.train_ds,train_split=True)
        self.val_ds = self.conf_perf_ds_from_csv(self.val_ds)
        self.test_ds = self.conf_perf_ds_from_csv(self.test_ds)

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

    def conf_perf_ds_from_csv(self, ds, train_split=False):
        ds = ds.cache()
        if train_split:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def conf_perf_ds_from_directory(self, ds, train_split=False):
        ds = ds.cache()
        if train_split:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds

    def build_model(self, model_name, trainable_layers=2, random_rotation=0.3, random_zoom=0.3, learning_rate=0.001):
        self.model_name = model_name
        self.trainable_layers = trainable_layers
        self.random_rotation = random_rotation
        self.random_zoom = random_zoom
        self.learning_rate=learning_rate

        assert self.model_name in {
            "VGG16", "ResNet50", "custom_1", "custom_2"}, "Choose a model among the following ones: 'VGG16', 'ResNet50', 'custom_1', 'custom_2"

        data_augmentation_layers = self.get_data_augmentation_layers()

        if self.model_name in {"ResNet50", "VGG16"}:
            if self.model_name == "VGG16":
                layer_model = self.get_vgg16_model()

            if self.model_name == "ResNet50":
                layer_model = self.get_resnet50_model()

            layer_model.trainable = False
            for layer in layer_model.layers[-self.trainable_layers:]:
                layer.trainable = True

            inputs = Input(shape=(self.img_height, self.img_width, 3))
            # Are not applied to validation and test dataset (made inactive, tensorflow handle it)
            x = data_augmentation_layers(inputs)
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

        if self.model_name == 'custom_1':
            self.model = models.Sequential([
                layers.InputLayer(input_shape=(
                    self.img_height, self.img_width, 3)),
                Rescaling(1./255),
                data_augmentation_layers,
                layers.Conv2D(8, 3, padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                layers.Conv2D(128, 3, padding='same', activation='relu'),
                layers.Flatten(),
                layers.Dropout(0.4),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        if self.model_name == 'custom_2':
            self.model = models.Sequential()
            self.model.add(Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)))
            self.model.add(data_augmentation_layers)

            self.model.add(layers.Conv2D(64, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(64, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            self.model.add(layers.Conv2D(128, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(128, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            self.model.add(layers.Conv2D(256, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(256, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(256, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Conv2D(512, (3, 3)))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            self.model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
            self.model.add(layers.Dense(4096))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dense(4096))
            self.model.add(layers.Activation('relu'))
            self.model.add(layers.Dropout(0.5))
            self.model.add(layers.Dense(8))
            self.model.add(layers.Activation('softmax'))

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

    def run(self, epochs=100, use_rlrp=False):
        self.epochs = epochs
        self.use_rlrp = use_rlrp

        assert self.model, "Run the build_model() method first"
        assert self.train_ds and self.val_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"

        es = EarlyStopping(monitor='val_loss', patience=20,
                           mode='min', restore_best_weights=True)
        callbacks = [es]

        if self.use_rlrp:
            rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.4,
                                     patience=3, min_lr=1e-8)
            callbacks.append(rlrp)

        self.history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            validation_data=self.val_ds,
            callbacks=callbacks,
            use_multiprocessing=True)

        return self.history

    def save_model(self, save_model_path=None):
        assert self.model, "Run the build_model() method first"
        assert self.history, "Run the run() method first"

        recorded_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.save_model_path = save_model_path

        if not self.save_model_path:
            self.save_model_path = os.path.join(
                '..','..','models', self.experiment_name, self.model_name, f"{recorded_time}-images_{self.image_count}-unfreeze_{self.trainable_layers}-batch_{self.batch_size}")

        self.model.save(self.save_model_path)

    def load_model(self, load_model_path):
        self.load_model_path = load_model_path
        self.model = models.load_model(self.load_model_path)

    def evaluate(self):
        assert self.test_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        assert self.model, "Run the build_model() method first"
        assert self.history or self.load_model_path, "Run the run() method first"
        self.results = self.model.evaluate(self.test_ds)
        return self.results

    def predict(self,image_path=None):
        assert self.test_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        assert self.model, "Run the build_model() method first"
        assert self.history or self.load_model_path, "Run the run() method first"

        if not image_path:
            return self.model.predict(self.test_ds)

        img = tf.io.read_file(image_path)
        img = self.decode_img(img)

        return self.model.predict(tf.expand_dims(img, axis=0))

    def plot_history(self):
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

    def plot_train_batch(self):
        assert self.train_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        image_batch, label_batch = next(iter(self.train_ds))

        plt.figure(figsize=(15, 15))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            plt.title(self.class_names[label.numpy() == 1][0])
            plt.axis("off")

    def plot_val_batch(self,make_prediction=False):
        assert self.val_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"

        image_batch, label_batch = next(iter(self.val_ds))

        if make_prediction:
            assert self.model, "Run the build_model() method first"
            assert self.history or self.load_model_path, "Run the run() method first"
            pred_batch = self.model.predict(image_batch.numpy())

        plt.figure(figsize=(15, 15))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            label = label_batch[i]
            plt.title(f"Truth: {self.class_names[label.numpy()==1][0]}")
            if make_prediction:
                plt.xlabel(
                    f"Pred: {' '.join([f'{self.class_names[a]}: {pred_batch[i][a]:0.2f}' for a in np.argsort(pred_batch[i])[::-1] if pred_batch[i][a] > 0.2])}")
                plt.xticks([])
                ax.set_xticks([])
                plt.yticks([])
                ax.set_yticks([])
            else:
                plt.axis("off")

    def plot_confusion_matrix(self):
        assert self.val_ds, "Run the create_dataset_from_directory() or create_dataset_from_csv() methods first"
        assert self.model, "Run the build_model() method first"
        assert self.history or self.load_model_path, "Run the run() method first"

        y_pred = []  # store predicted labels
        y_true = []  # store true labels

        # iterate over the dataset
        for image_batch, label_batch in self.val_ds:
            # append true labels
            y_true.append(np.argmax(label_batch, axis=1))
            # compute predictions
            preds = self.model.predict(image_batch)
            # append predicted labels
            y_pred.append(np.argmax(preds, axis=- 1))

        # convert the true and predicted labels into tensors
        correct_labels = tf.concat([item for item in y_true], axis=0)
        predicted_labels = tf.concat([item for item in y_pred], axis=0)

        conf_mat = tf.math.confusion_matrix(correct_labels, predicted_labels)
        conf_mat = conf_mat.numpy() / conf_mat.numpy().sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        sns.heatmap(conf_mat, annot=True, cmap="Blues", ax=ax, cbar=False,
                    xticklabels=self.class_names, yticklabels=self.class_names, fmt='.2%')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
        ax.set_ylabel("Labels")
        ax.set_xlabel("Prediction")
