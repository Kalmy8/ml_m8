import numpy as np

def epanch_kernel (x : np.array, y : np.array, bandwidth):
  r = np.sum(np.linalg.norm(x-y)) / bandwidth
  return 0 if r >=1 else 0.75 * (1 - np.square(r))

def gaussian_kernel (x : np.array, y : np.array, bandwidth):
  r = np.sum(np.linalg.norm(x-y)) / bandwidth
  return 0.4 * np.exp(-0.5 * np.square(r))

class Parzen:
  '''
  Similar to kNN classifier, but uses kernels to calculate distances to neighbours
  See more : https://youtu.be/H4P6u4918fI?si=L6P0o5eOLRHKRCW_
  '''
  def __init__(self, kernel_func, bandwidth):
    self.kernel = kernel_func
    self.X_fitted = None
    self.y_fitted = None
    self.h = bandwidth

  # Сеттер
  def fit(self, X_train : np.array, y_train : np.array):
    self.X_fitted = X_train
    self.y_fitted = y_train

  def _calc_distances(self, X, X_to_compare):
    probs = []
    for sample in X:
      proba = self.kernel(sample, X_to_compare, self.h)
      probs.append(proba)
    return np.array(probs)


  def predict(self, X):
      feature_labels = np.unique(self.y_fitted)
      answer = np.zeros(shape = (X.shape[0], feature_labels.size)) # Форма для ответов размера кол-во наблюдений X кол-во возможных классов

      for label in feature_labels: # Для каждой метки класса
        ix = (self.y_fitted == label).nonzero()[0] # Посмотрим индексы тех экземпляров, кто ему принадлежит
        probs = self._calc_distances(X, self.X_fitted[ix, :]) / ix.size # Посчитаем дистанции между этими и тестовыми экземплярами
        answer[:, label] = probs # Для каждого тестового экземпляра в соответствующий столбец занесем вероятность


      return answer