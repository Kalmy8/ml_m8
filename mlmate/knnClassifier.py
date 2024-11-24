import pandas as pd
import numpy as np


class knnClassifier:
  ''' Powered by pandas, works by euqlidian distance'''

  def __init__(self, X_train, y_train, k):
    self.k = k
    self.X_train = X_train
    self.y_train = y_train

  # Функция, вычисляющая расстояние между двумя наблюдениями
  def euqlid_distance(nabl1: pd.Series, nabl2: pd.Series) -> float:
      feature_distances = []
      for i in range(nabl1.size):
          feature_distances.append((nabl1[i] - nabl2[i]) ** 2)

      return np.sqrt(np.sum(feature_distances))

  # Функцию, которая согласно расстояния подбирает k ближайших соседей, и отдает обратно их индексы в предоставленном dataset
  def _find_neighbours(self, nabl: pd.Series, X_train: pd.DataFrame, k: int) -> list:
      distances = {}
      for index, row in X_train.iterrows():  # Для каждого наблюдения из обучающей выборки
          distances[index] = (knnClassifier.euqlid_distance(nabl, row))  # Посчитаем расстояние до заданного наблюдения

      distances = pd.Series(distances)
      distances = distances.sort_values().iloc[:k]

      ix = distances.index
      return ix

  def predict(self, X_test):
    predictions = {}

    for ix, nabl in X_test.iterrows(): # Для каждого неопознанного ириса

      nearest_neighbours_ix = self._find_neighbours(nabl, self.X_train, self.k) # Находим ближайших k соседей в обучающем множестве (гербарии), забираем их индексы
      mode_value = self.y_train.loc[nearest_neighbours_ix].mode()[0] # По индексам забираем модальное значение
      predictions[ix] = mode_value

    return pd.Series(predictions)
