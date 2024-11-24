import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
from collections import Counter
import csv
from itertools import chain

class FP_Node:
    def __init__(self, suffix: str, prefix: list[object]):
        self.prefix = prefix
        self.suffix = suffix
        self.children = []
        self.parent = prefix[-1] if len(prefix) > 0 else []
        self.counter = 0

    def forward_pass(self, transaction, header_table):
        self.counter += 1

        # If recieved transaction is empty, we are at the leaf node
        if len(transaction) >= 1:
            # Gets first item from passed transaction
            item = transaction.pop(0)

            # Any of your children contains this item?
            next_node = [child for child in self.children if child.suffix == item]
            if len(next_node) == 1:
                next_node[0].forward_pass(transaction, header_table)
            else:
                new_prefix = self.prefix + [self]
                new_node = FP_Node(item, new_prefix)

                # Whenether new node is born, the according note is being putted in the main table
                header_table[item].append(new_node)
                self.children.append(new_node)
                new_node.forward_pass(transaction, header_table)

class FP_Tree:
    def __init__(self):
        self.header_table = defaultdict(list)
        self.root = FP_Node('root', [])
        self.most_frequent_sets = defaultdict(list)

    def update_tree(self, transactions : list):
        for transaction in transactions:
            self.root.forward_pass(transaction.copy(), self.header_table)

    @staticmethod
    def _get_parents_freqs(data : list[FP_Node]):
        # Extract all the item's ancestors, count their frequencies
        parents_freq = defaultdict(lambda: 0)
        for node in data:
            for parent_nodes in node.prefix:
                parents_freq[parent_nodes.suffix] += node.counter

        parents_freq.pop('root', None)
        return dict(parents_freq)

    def _reduce_itemsets(self):
        """
        Reduce item sets by removing subsets of more frequent item sets.
        :return: Reduced dictionary of item sets
        """
        # Sort itemsets by frequency in descending order and by length of the itemset in descending order
        for suffix, itemset in self.most_frequent_sets.items():
            sorted_itemsets = sorted(itemset, key = lambda x: (-len(x[0]), -x[1]))

            reduced_itemsets = {}
            for itemset, frequency in sorted_itemsets:
                add_itemset = True

                for items, freq in reduced_itemsets.items():
                    if frequency <= freq and set(itemset).issubset(set(items)):
                        add_itemset = False

                if add_itemset:
                    reduced_itemsets[tuple(itemset)] = frequency

            self.most_frequent_sets[suffix] = reduced_itemsets

    def mine_tree(self, data: list, path: tuple, min_supp: float):
        parents_freq = FP_Tree._get_parents_freqs(data)
        filtered_freqs =  dict((k, v) for k, v in parents_freq.items() if v >= min_supp)

        if len(filtered_freqs) > 0:
            for new_suffix in filtered_freqs.keys():
                new_data = list(set(chain.from_iterable([node.prefix for node in data])))
                new_data = list(filter(lambda node: node.suffix == new_suffix, new_data))
                new_path = (new_suffix,) + path
                self.most_frequent_sets[path[-1]].append((new_path, filtered_freqs[new_suffix]))
                self.mine_tree(new_data,new_path, min_supp)

    def get_most_frequent_sets(self, min_supp : float):
        nodes_count = len(list(chain.from_iterable(self.header_table.values())))
        min_supp = min_supp * nodes_count

        # For every item, get all of the nodes
        for suffix, nodes in self.header_table.items():

            # Refresh nodes_counters
            nodes_recounted = nodes.copy()

            for node in nodes_recounted:
                for ancestor in node.prefix:
                    ancestor.counter = 0

            for node in nodes_recounted:
                for ancestor in node.prefix:
                    ancestor.counter += node.counter

            self.mine_tree(nodes_recounted, (suffix, ), min_supp)

        # Reduce answer
        self._reduce_itemsets()
        return self.most_frequent_sets

class Data_loader():
    @staticmethod
    def __call__(filename: str):
        x = Data_loader.load_data(filename)
        freqs = Data_loader.count_frequencies(x)
        x = Data_loader.sort_and_clean_transactions(x, freqs)
        return x

    @staticmethod
    def load_data(filename: str):
        transactions = []
        with open(filename) as database:
            for row in csv.reader(database):
                transactions.append(row)

        return transactions

    @staticmethod
    def count_frequencies(transactions):
        freq_dict = defaultdict(lambda: 0)
        for transaction in transactions:
            for item in transaction:
                freq_dict[item] += 1

        freq_dict = dict((item, support) for item, support in freq_dict.items())

        return freq_dict

    @staticmethod
    def sort_and_clean_transactions(transactions, freqs):
        transactions_sorted = []
        for transaction in transactions:
            transaction = list(filter(lambda item: item in freqs, transaction))
            transaction.sort(key=lambda item: freqs[item], reverse=True)
            transactions_sorted.append(transaction)
        return transactions_sorted


def calculate_confidence(itemsets):
    """
    Calculate confidence scores for association rules from frequent itemsets.

    :param itemsets: Dictionary with itemsets as keys (tuples) and their support counts as values
    :return: Dictionary of association rules with their confidence scores
    """
    rules_confidence = {}

    for itemset, support_count in itemsets.items():
        if len(itemset) > 1:
            itemset_list = list(itemset)
            n = len(itemset_list)
            # Generate all non-empty subsets of the itemset
            for i in range(1, 1 << n):
                antecedent = tuple(sorted(itemset_list[j] for j in range(n) if (i & (1 << j))))
                if antecedent and len(antecedent) < len(itemset):
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    if antecedent in itemsets:
                        confidence = support_count / itemsets[antecedent]
                        rules_confidence[(antecedent, consequent)] = confidence

    return rules_confidence

def main():
    data_path = 'toy_datasets/Market_Basket_Optimisation.csv'
    data_loader = Data_loader()
    preprocessed_data = data_loader(data_path)
    tree = FP_Tree()
    tree.update_tree(preprocessed_data)
    sets = tree.get_most_frequent_sets(0.005)
    #calculate_confidence(sets.values())
    print(sets)

if __name__ == '__main__':
    main()

