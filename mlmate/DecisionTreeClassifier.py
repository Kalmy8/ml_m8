import numpy as np
import pandas as pd

class Node:
    """Node class used for DTC building"""
    def __init__(self, dataset: pd.DataFrame, target_name: str, parent_tree, depth):
        # Характеристики данных
        self.dataset = dataset
        self.target_name = target_name
        self.n_samples = [(dataset[target_name] == 0).sum(), (dataset[target_name] == 1).sum()]
        self.main_class = 1 if self.n_samples[1] > self.n_samples[0] else 0

        # Характеристики ноды
        self.depth = depth
        self.parent_tree = parent_tree
        self.is_leaf = False
        self.gini = None
        self.question = (None, None)
        self.children = (None, None)

        # Node init_functions
        self._set_question_and_gini()
        self._create_children()

    # Разбиваем исходное множество по признаку (split feature) используя порог (thr), считаем взвешенную gini-метрику для такого разбиения
    def _math_gini(self, split_feature: str, thr: float) -> float:

        true_part = self.dataset[self.dataset[split_feature] >= thr]
        false_part = self.dataset[self.dataset[split_feature] < thr]

        gini_true = 1 - np.square(true_part[self.target_name].sum() / true_part.shape[0]) - np.square(
            (true_part[self.target_name] == 0).sum() / true_part.shape[0])
        gini_false = 1 - np.square(false_part[self.target_name].sum() / false_part.shape[0]) - np.square(
            (false_part[self.target_name] == 0).sum() / false_part.shape[0])
        gini_weighted = (true_part.shape[0] * gini_true + false_part.shape[0] * gini_false) / self.dataset.shape[0]

        return gini_weighted

    # Обертка над _math_gini для работы с cat/num. признаками, возвращает значение метрики Gini Impurity и порог, при котором оно достигается
    def _calc_gini(self, split_feature):

        # Если признак категориальный, то разбиение будем вести с помощью порога в 0.5
        if self.dataset[split_feature].dtype.name == 'category':
            thr = 0.5
            return self._math_gini(split_feature, thr), thr

        # Для численных признаков нужно подобрать целый пласт порогов, на основе которых можно провести разбиение
        else:
            best_gini = 1
            best_thr = None

            thrs = self.dataset[split_feature].sort_values(ascending=True).drop_duplicates().rolling(2).mean()[1:]

            for thr in thrs:
                gini = self._math_gini(split_feature, thr)
                if gini <= best_gini:
                    best_gini = gini
                    best_thr = thr

            return best_gini, best_thr

    # Перебирает все возможные разбиения и выбирает лучшее, устанавливает self.gini и self.question
    def _set_question_and_gini(self):
        best_gini = 1

        for feature in self.dataset.columns.drop(self.target_name):

            gini_criteria, threshold = self._calc_gini(feature)

            if gini_criteria <= best_gini:
                best_gini = gini_criteria
                self.question = (feature, threshold)
                self.gini = gini_criteria

                if gini_criteria == 0:
                    self.is_leaf = True

    # Порождает детей на основе лучшего разбиения, если нода — не лист
    def _create_children(self):

        true_part = self.dataset[self.dataset[self.question[0]] >= self.question[1]]
        false_part = self.dataset[self.dataset[self.question[0]] < self.question[1]]

        # Проверяем, не нарушаются ли ограничения
        if self.parent_tree.restrictions:
            if sum(self.n_samples) <= self.parent_tree.restrictions['min_samples_split'] or \
                    true_part.shape[0] <= self.parent_tree.restrictions['min_samples_leaf'] or \
                    false_part.shape[0] <= self.parent_tree.restrictions['min_samples_leaf'] or \
                    self.depth >= self.parent_tree.restrictions['max_depth']:
                self.is_leaf = True

        # Если эта нода — не лист, то она порождает детей:
        if self.is_leaf == False:
            self.parent_tree.depth = self.depth + 1

            # Создаём дочерние ноды:
            true_child = Node(true_part, self.target_name, self.parent_tree, self.depth + 1)
            false_child = Node(false_part, self.target_name, self.parent_tree, self.depth + 1)
            self.children = (true_child, false_child)

    # Forward pass
    def forward_pass(self, X: pd.DataFrame):
        if self.is_leaf == False:
            if X.size != 0:
                feature, th = self.question
                true_subpart = X[X[feature] >= th]
                false_subpart = X[X[feature] < th]

                self.children[0].forward_pass(true_subpart)
                self.children[1].forward_pass(false_subpart)

        else:
            self.parent_tree.prediction_result[X.index] = self.main_class


class DTC():
    """Powered by pandas"""
    def __init__(self, restrictions: dict = None):
        self.restrictions = restrictions
        self.depth = 1
        self.root_node = None
        self.prediction_result = None
        self.features_seen = None

    # Метод для обучения
    def fit(self, dataset: np.array, target_name: str):
        self.root_node = Node(dataset, target_name, self, 1)  # Далее ноды будут инициализировать друг друга автоматически
        self.features_seen = dataset.drop(target_name, axis=1).columns

    # Метод для предсказания
    def predict(self, X: np.array):
        self.prediction_result = pd.Series(index=X.index)
        self.root_node.forward_pass(X)

        return self.prediction_result
