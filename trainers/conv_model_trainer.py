import tensorflow as tf
import numpy as np


class ConvModelTrainer:

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def train(self, train, valid):
        optimizer = tf.optimizers.Adam(learning_rate=self.config.trainer.learning_rate)
        for i in range(self.config.trainer.num_iteration):
            x_batch, y_true_batch, _, cls_batch = train.next_batch(self.config.glob.batch_size)
            with tf.GradientTape() as tape:
                # Get the predictions
                y_train_pred = self.model.run(x_batch)
                # Calc the loss
                current_loss = ConvModelTrainer.loss_function(y_train_pred, y_true_batch)
                # Get the gradients
                grads = tape.gradient(current_loss, self.model.trainable_variables())
                # Update the weights
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables()))
            #Show grogress
            if i % self.config.trainer.display_num == 0:
                y_val_preds = []
                for val_image in valid.images:
                    y_val_preds.append(self.model.run(np.expand_dims(val_image, axis=0)))
                y_train_preds = []
                for train_image in train.images:
                    y_train_preds.append(self.model.run(np.expand_dims(train_image, axis=0)))
                y_val_pred = np.vstack(y_val_preds)
                y_train_pred = np.vstack(y_train_preds)
                ConvModelTrainer.show_progress(i, train.labels, y_train_pred, valid.labels, y_val_pred)


    @staticmethod
    def show_progress(epoch, train_true, train_pred, valid_true, valid_pred):
        train_acc = ConvModelTrainer.accuracy(train_pred, train_true)
        val_acc = ConvModelTrainer.accuracy(valid_pred, valid_true)
        train_loss = tf.reduce_mean(ConvModelTrainer.loss_function(train_pred, train_true))
        val_loss = tf.reduce_mean(ConvModelTrainer.loss_function(valid_pred, valid_true))
        msg = "Training Epoch {0} --- training Accuracy: {1:.4f}, training Losss: {2:.4f} --- validation Accuracy: {3:.4f},  validation Loss: {4:.4f}"
        print(msg.format(epoch + 1, train_acc, train_loss, val_acc, val_loss))

    @staticmethod
    def loss_function(y_pred, y_true):
        return tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true), logits=y_pred)

    @staticmethod
    def accuracy(y_pred, y_true):
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
