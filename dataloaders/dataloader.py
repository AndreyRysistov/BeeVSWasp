import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from dataloaders.dataset import DataSet


class DataLoader:

    def __init__(self, path, config):
        self.config = config
        self.data_path = path
        self.classes = os.listdir(path)

    def create_datasets(self):
        images, labels, img_names, cls = self._load_train(self.data_path)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
        validation_size = int(self.config.dataloader.validation_size * images.shape[0])
        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_img_names = img_names[:validation_size]
        validation_cls = cls[:validation_size]
        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_img_names = img_names[validation_size:]
        train_cls = cls[validation_size:]

        train = DataSet(train_images, train_labels, train_img_names, train_cls)
        valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

        return train, valid

    def _load_train(self, train_path):
        images = []
        labels = []
        img_names = []
        cls = []
        print('Going to read training images')
        for fields in self.classes:
            index = self.classes.index(fields)
            print('Now going to read {} files (Index: {})'.format(fields, index))
            path = os.path.join(train_path, fields, '*g')
            files = glob.glob(path)
            for fl in files:
                image = self._read_image(fl)
                images.append(image)
                label = np.zeros(len(self.classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                cls.append(fields)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        cls = np.array(cls)

        return images, labels, img_names, cls

    def _read_image(self, file):
        image = tf.io.read_file(file)
        image = tf.image.decode_jpeg(image, self.config.glob.image_channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.config.glob.image_size)
        return image








