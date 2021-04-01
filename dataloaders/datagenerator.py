import tensorflow as tf
import os


class DataLoader:

    def __init__(self, config):
        self.config = config
        self.path = './datasets/data'
        self.class_names = os.listdir(self.path)
        self.ds = self.set_generator()

    def get_data(self):
        train, val = self.split_data()
        return train, val

    def split_data(self):
        data = self.ds.shuffle(1000)
        labeled_all_length = [i for i,_ in enumerate(data)][-1] + 1
        train_size = int(self.config.dataloader.train_size * labeled_all_length)
        val_size = int(self.config.dataloader.validation_size * labeled_all_length)
        print('Train: ', train_size)
        print('Validation :', val_size)
        train = data.take(train_size).cache().repeat().batch(self.config.glob.batch_size)
        val = data.skip(train_size).cache().batch(val_size)
        return train, val

    def set_generator(self):
        filenames = tf.data.Dataset.list_files("./datasets/data/*/*.jpg")
        ds = filenames.map(self.process_path).shuffle(buffer_size=1000)
        return ds

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return tf.cast(one_hot, 'float64', name=None)

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, self.config.glob.image_channels)
        img = tf.image.convert_image_dtype(img, tf.float32) / 255
        return tf.image.resize(img, self.config.glob.image_size)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label