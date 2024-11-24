import numpy as np
import tensorflow_datasets as tfds

class MaxPooling:
    def __init__(self, pool_shape: tuple = (2, 2), stride: tuple = (1, 1)):
        self.pool_shape = pool_shape
        self.stride = stride
        # Memory for backprop
        self.last_input = None

    def __call__(self, in_channels: np.array):

        self.last_input = in_channels
        input_shape = in_channels.shape

        output_shape = (input_shape[0],
                        (input_shape[1] - self.pool_shape[0]) // self.stride[0] + 1,
                        (input_shape[2] - self.pool_shape[1]) // self.stride[1] + 1
                        )

        output_array = np.zeros(output_shape)

        for z in range(input_shape[0]):
            for i in range(0, input_shape[1] - self.pool_shape[0] + 1, self.stride[0]):
                for j in range(0, input_shape[2] - self.pool_shape[1] + 1, self.stride[1]):
                    window = in_channels[z, i:i + self.pool_shape[0], j:j + self.pool_shape[1]]
                    output_array[z, i // self.stride[0], j // self.stride[1]] = np.max(window)

        return output_array

    def backward_pass(self, dl_dout):
        dX = np.zeros_like(self.last_input)  # Градиент функции потерь по входу слоя

        for z in range(self.last_input.shape[0]): # По всем каналам
            for i in range(0, self.last_input.shape[1] - self.pool_shape[0] + 1, self.stride[0]):
                for j in range(0, self.last_input.shape[2] - self.pool_shape[1] + 1, self.stride[1]):
                    window = self.last_input[z, i:i + self.pool_shape[0], j:j + self.pool_shape[1]]
                    max_val = np.max(window)
                    dX[z, i:i + self.pool_shape[0], j:j + self.pool_shape[1]] += (window == max_val) * dl_dout[
                        z, i // self.stride[0], j // self.stride[1]]

        return dX


class Flatten:
    def __init__(self):
        # Memory for backprop
        self.last_input_shape = None

    def __call__(self, in_channels: np.array):
        self.last_input_shape = in_channels.shape
        return np.ravel(in_channels).reshape(-1, 1)

    def backward_pass(self, dl_dout: np.array):
        return dl_dout.reshape(self.last_input_shape)


class Conv2d():
    def __init__(self, n_filters, kernel_shape: tuple = (3, 3), stride=(1, 1), padding=None, activation_func=None):

        self.n_filters = n_filters
        self.kernel_shape = kernel_shape
        self.kernels = None
        self.stride = stride
        self.padding = padding  # TODO
        self.activation_func = activation_func
        self.biases = np.zeros(shape=(n_filters, 1, 1))

        # Memory for backprop
        self.last_input = None

    def __call__(self, in_channels):
        input_shape = in_channels.shape

        if self.kernels is None:
            self.kernels = np.random.random(size=(self.n_filters, input_shape[0], *self.kernel_shape))

        self.last_input = in_channels

        stride_x, stride_y = self.stride
        output_shape = (self.n_filters,
                        (input_shape[1] - self.kernel_shape[0]) // stride_x + 1,
                        (input_shape[2] - self.kernel_shape[1]) // stride_y + 1)

        output_array = np.zeros(output_shape)

        for z, kernel in enumerate(self.kernels):  # Для каждого ядра
            for i in range(0, input_shape[1] - self.kernel_shape[0] + 1, stride_x):
                for j in range(0, input_shape[2] - self.kernel_shape[1] + 1, stride_y):
                    window = in_channels[:, i:i + self.kernel_shape[0], j:j + self.kernel_shape[1]]
                    output_array[z, i // stride_x, j // stride_y] = np.sum(window * kernel)

        if self.activation_func:
            output_array = self.activation_func(output_array + self.biases)

        else:
            output_array = output_array + self.biases

        return output_array

    def backward_pass(self, dl_dout, lr):

        dZ = dl_dout  # Градиент функции потерь по выходу слоя
        dW = np.zeros_like(self.kernels)  # Градиент функции потерь по ядрам свертки
        dX = np.zeros_like(self.last_input)  # Градиент функции потерь по входу слоя
        dB = np.sum(dZ, axis=(1, 2), keepdims=True)  # Градиент функции потерь по смещениям

        for z in range(self.kernels.shape[0]): # Для каждого фильтра
            for c in range(self.kernels.shape[1]):  # Для каждого входного канала
                for i in range(0, self.last_input.shape[1] - self.kernel_shape[0] + 1, self.stride[0]):
                    for j in range(0, self.last_input.shape[2] - self.kernel_shape[1] + 1, self.stride[1]):
                        window = self.last_input[c, i:i + self.kernel_shape[0], j:j + self.kernel_shape[1]]
                        dW[z, c, :, :] += window * dZ[z, i // self.stride[0], j // self.stride[1]]
                        dX[c, i:i + self.kernel_shape[0], j:j + self.kernel_shape[1]] = np.add(
                            dX[c, i:i + self.kernel_shape[0], j:j + self.kernel_shape[1]],
                            self.kernels[z, c, :, :] * dZ[z, i // self.stride[0], j // self.stride[1]],
                            casting='unsafe'
                            )

        self.kernels -= dW * lr
        self.biases -= dB * lr

        return dX


class Linear():
    def __init__(self, output_dim, activation_function):
        self.W = None
        self.b = np.zeros(shape=(output_dim, 1))
        self.activation_function = activation_function
        self.output_dim = output_dim

        self.last_output = None
        self.last_input = None

    # Магический метод __call__
    def __call__(self, input_vector: np.array):
        # When seeing input_dim for first time, can specify W matrix shape
        if self.W is None:
            self.W = np.random.random(size=(self.output_dim, input_vector.shape[0]))

        self.last_input = input_vector
        self.last_output = self.activation_function(self.W @ input_vector.reshape(-1, 1) + self.b)
        return self.last_output

    def backward_pass(self, dl_dout: np.array, lr):
        # Gradients of loss against weights/biases/input
        d_l_d_w = dl_dout @ self.last_input.T
        d_l_d_b = dl_dout

        d_l_d_input = self.W.T @ dl_dout
        # https: // web.eecs.umich.edu / ~justincj / teaching / eecs442 / notes / linear - backprop.html
        # update weights/biases
        self.W -= lr * d_l_d_w
        self.b -= lr * d_l_d_b

        return d_l_d_input


class Softmax():
    def __init__(self, output_dim, activation_function):
        self.W = None
        self.b = np.zeros(shape=(output_dim, 1))
        self.activation_function = activation_function
        self.output_dim = output_dim

        # Sort of memory for backpropagation
        self.Z = None

    def __call__(self, input_vector: np.array):
        # When seeing input_dim for first time, can specify W matrix shape
        if self.W is None:
            self.W = np.random.random(size=(self.output_dim, input_vector.shape[0]))

        self.Z = self.activation_function(self.W @ input_vector.reshape(-1, 1) + self.b)
        exp = np.exp(self.Z)
        return (exp / np.sum(exp, axis=0))  # Normalization

    def backward_pass(self, dl_dout: np.array, lr):
        # We know only 1 element of d_l_d_out will be nonzero
        for i, gradient in enumerate(dl_dout):
            if (gradient == 0):
                continue

            # Calc last_exp
            Z_exp = np.exp(self.Z)
            # Calc last_sum
            S = np.sum(Z_exp)

            # gradients of out[i] against self.Z
            d_out_d_Z = -Z_exp[i] * Z_exp / (S ** 2)
            d_out_d_Z[i] = Z_exp[i] * (S - Z_exp[i]) / (S ** 2)

            # Gradients of Z against weights/biases/input
            d_Z_d_w = self.Z
            d_Z_d_b = 1
            d_Z_d_input = self.W

            # Gradients of loss against Z
            d_l_d_Z = gradient * d_out_d_Z

            # Gradients of loss against weights/biases/input
            d_l_d_w = d_Z_d_w.T @ d_l_d_Z
            d_l_d_b = d_l_d_Z * d_Z_d_b
            d_l_d_input = d_Z_d_input.T @ d_l_d_Z

            # update weights/biases
            self.W -= lr * d_l_d_w
            self.b -= lr * d_l_d_b

            return d_l_d_input


def sigmoid(x: np.array) -> np.array:
    return (1 + np.exp(-x)) ** -1


def dsigmoid(x: np.array) -> np.array:
    return sigmoid(x) * (1 - sigmoid(x))


class CNN():
    def __init__(self):
        self.c1 = Conv2d(n_filters=32)
        self.max_pool = MaxPooling(pool_shape=(4, 4), stride=(4, 4))
        self.flat = Flatten()
        self.dense1 = Linear(output_dim=10, activation_function=sigmoid)
        self.softmax = Softmax(output_dim=10, activation_function=sigmoid)

    def predict(self, X: np.array):
        X = self.c1(X)
        X = self.max_pool(X)
        X = self.flat(X)
        X = self.dense1(X)
        X = self.softmax(X)

        return X

    def _backprop(self, dl_dout, lr):
        X = self.softmax.backward_pass(dl_dout, lr)
        X = self.dense1.backward_pass(X, lr)
        X = self.flat.backward_pass(X)
        X = self.max_pool.backward_pass(X)
        X = self.c1.backward_pass(X, lr)

    def train(self, train_data, train_labels, num_epochs=2, lr=1e-2):
        for i in range(num_epochs):
            print('----EPOCH %d ---' % (i + 1))

            indexes = np.arange(train_labels.shape[0])  # shuffle the training data
            permutation = np.random.permutation(indexes)
            train_image = train_data[permutation]
            train_label = train_labels[permutation]

            loss = 0
            num_correct = 0

            for i, (im, label) in enumerate(zip(train_data, train_labels)):
                im = (im / 255) - 0.5  # RGB Normalize [-0.5, 0.5]
                # print stats every 100 steps
                if i % 100 == 0:
                    print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (
                    i + 1, loss / 100, num_correct))
                    loss = 0
                    num_correct = 0

                # make prediction
                prediction = self.predict(im)
                loss = -np.log(prediction[label])
                acc = 1 if (np.argmax(prediction) == label) else 0

                loss += loss  # память для усреднения
                num_correct += acc

                # calculate initial gradient
                gradient = np.zeros(10)
                gradient[label] = -1 / prediction[label] # -1/x

                # update params
                self._backprop(gradient, lr)


def main():
    image, label = tfds.load('cifar10', split='all', download=False, batch_size=-1,
                             as_supervised=True)
    image = np.moveaxis(image, -1, 1) # cifar 10 в формате высота x ширину x каналы, переделаю в каналы Х высота Х ширина
    label = label.numpy()

    mynet = CNN()
    mynet.train(image, label)


if __name__ == '__main__':
    #main()
