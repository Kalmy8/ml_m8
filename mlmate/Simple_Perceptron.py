import numpy as np

def sigmoid(x : np.array) -> np.array:
  return (1 + np.exp(-x))**-1

def dsigmoid(x : np.array) -> np.array:
  return sigmoid(x)*(1-sigmoid(x))

def MSE(y_pred : np.array, y_true : np.array) -> np.array:
  return np.sum(0.5 * np.square(y_pred-y_true))

def dMSE(y_pred : np.array, y_true : np.array) -> np.array:
  return np.sum(y_pred-y_true)

class Simple_Perceptron:
  def __init__(self, struct : tuple, activation, dactivation, loss, dloss):
    # Structure: depth, neurons amount
    self.depth = len(struct) - 1

    # Activations, Z, weights and biases matrices, their deltas
    self.A = [None] * (self.depth + 1)
    self.Z = [None] * self.depth
    self.W = self._init_weights(struct)
    self.B = self._init_biases(struct)

    self.dW = [np.zeros_like(w) for w in self.W]
    self.dB = [np.zeros_like(b) for b in self.B]
    self.dZ = [np.zeros_like(z) for z in self.Z]

    # Activate and loss functions
    self.activ_func = activation
    self.dactiv_func = dactivation
    self.loss_func = loss
    self.dloss_func = dloss

  # Glorot Method
  def _init_weights(self, struct : tuple) -> list:

    W = [None] * self.depth
    input_size = struct[0]
    output_size = struct[-1]
    limit = np.sqrt(6/(input_size + output_size))

    for i in range(self.depth):
      w_size = (struct[i+1], struct[i])
      W[i] = np.random.uniform(-limit, limit, size = w_size)

    return  W

  # Normal dist
  def _init_biases(self, struct : tuple) -> list:
    B = [None] * self.depth
    for i in range(self.depth):
      b_size = (struct[i+1],1)
      B[i] = np.random.normal(size = b_size)
    return B

  # Perform a forward pass
  def _forward_pass(self, X):
    self.A[0] = X

    for i in range(self.depth):
      self.Z[i] = np.matmul(self.W[i],self.A[i]) + self.B[i]

      # На последнем слое функцию активации не применяем
      if i != self.depth - 1:
        self.A[i+1] = self.activ_func(self.Z[i])
      else:
        self.A[i+1] = self.Z[i]

  # Perform a backward pass - computed dZ,dW,dB matrices
  def _backprop(self, y):
    # Сначала посчитаем последний слой
    self.dZ[-1] = self.dloss_func(self.A[-1], y)
    self.dW[-1] = self.dZ[-1] @ self.A[self.depth - 1].T
    self.dB[-1] = self.dZ[-1]

    for i in range(self.depth-2, -1, -1):
      self.dZ[i] = self.W[i+1].T @ self.dZ[i+1] * self.dactiv_func(self.Z[i])
      self.dW[i] = self.dZ[i] @ self.A[i].T
      self.dB[i] = self.dZ[i]

  # Performs a forward pass and slices last layer activation
  def predict(self, X):
    X = np.array(X, ndmin = 2)
    self._forward_pass(X)
    y_pred = self.A[-1]
    return y_pred

  # Calculates loss for given example
  def _calculate_loss(self, X, y):
    y_pred = self.predict(X)
    return self.loss_func(y_pred, y)

  # Updates W,B with dW, dB matrices
  def _update_params(self, lr):
    for i in range(self.depth):
      self.W[i] -= lr * self.dW[i]
      self.B[i] -= lr * self.dB[i]

  #Performs training
  def train(self, X, y, epochs = 10, lr= 1e-2):
    loss_history = np.array([])

    dataset = list(zip(X, y))
    for i in range(epochs):
      #np.random.shuffle(dataset)

      for X, y in dataset:
        X = np.array(X, ndmin = 2)
        self._forward_pass(X)
        self._backprop(y)
        self._update_params(lr)

      loss_history = np.append(loss_history, self._calculate_loss(X,y))


    return loss_history