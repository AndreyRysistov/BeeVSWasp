from models.layers import *


class ConvModel:

    def __init__(self, config):
        self.config = config
        self.pool_size = self.config.model.pool_size
        self.dropout = self.config.model.dropout
        self.nclasses = self.config.glob.nclasses
        n = int((self.config.glob.image_size[0] / (self.pool_size * 2))**2)
        print(n)
        self.shapes = [
            # 1 conv layer
            [
                self.config.model.filter_size,
                self.config.model.filter_size,
                self.config.glob.image_channels,
                self.config.model.shapes.conv1
             ],
            # 2 conv layer
            [
                self.config.model.filter_size,
                self.config.model.filter_size,
                self.config.model.shapes.conv1,
                self.config.model.shapes.conv2
             ],
            # fully connected 1
            [
                n * self.config.model.shapes.conv2,
                self.config.model.shapes.fc1
             ],
            # fully connected 2
            [
                self.config.model.shapes.fc1,
                self.config.model.shapes.fc2
             ],
            # output layer
            [
                self.config.model.shapes.fc2,
                self.config.glob.nclasses
            ]
        ]
        self.weights = []
        for i in range(len(self.shapes)):
            print(self.shapes[i])
            self.weights.append(get_tfVariable(self.shapes[i], 'weight{}'.format(i)))

        self.bias = []
        for i in range(len(self.shapes)):
            self.bias.append(get_tfVariable([1, self.shapes[i][-1]], 'bias{}'.format(i)))

    def run(self, x_input):
        conv1 = conv_layer(x_input, self.weights[0], self.bias[0])
        pool1 = maxPool_layer(conv1, poolSize=self.pool_size)
        conv2 = conv_layer(pool1, self.weights[1], self.bias[1])
        pool2 = maxPool_layer(conv2, poolSize=self.pool_size)
        flat1 = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
        fully1 = tf.nn.relu(fullyConnected_layer(flat1, self.weights[2], self.bias[2]))
        fully1_dropout = tf.nn.dropout(fully1, rate=self.dropout)
        fully2 = tf.nn.relu(fullyConnected_layer(fully1_dropout, self.weights[3], self.bias[3]))
        fully2_dropout = tf.nn.dropout(fully2, rate=self.dropout)
        out = fullyConnected_layer(fully2_dropout, self.weights[4], self.bias[4])
        return out

    def trainable_variables(self):
        return self.weights + self.bias
