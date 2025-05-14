from itertools import combinations
from collections import Counter


class AprioriAlgorithm:
    def __init__(self, min_support=0.2):
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.transactions = []
    
    def fit(self, transactions):
        """
        Знаходить всі часті набори елементів
        
        Args:
            transactions: список транзакцій, де кожна транзакція - це список елементів
        """
        self.transactions = [set(transaction) for transaction in transactions]
        n_transactions = len(self.transactions)
        min_support_count = int(self.min_support * n_transactions)
        
        item_counts = Counter()
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        frequent_1_itemsets = {
            frozenset([item]): count 
            for item, count in item_counts.items() 
            if count >= min_support_count
        }
        
        self.frequent_itemsets[1] = frequent_1_itemsets
        
        k = 2
        while self.frequent_itemsets.get(k-1):
            candidates = self._generate_candidates(k)
            frequent_k_itemsets = {}
            
            for candidate in candidates:
                count = self._count_support(candidate)
                if count >= min_support_count:
                    frequent_k_itemsets[candidate] = count
            
            if frequent_k_itemsets:
                self.frequent_itemsets[k] = frequent_k_itemsets
                k += 1
            else:
                break
        
        return self.frequent_itemsets
    
    def _generate_candidates(self, k):
        """Генерує кандидатів для k-елементних наборів"""
        prev_itemsets = list(self.frequent_itemsets[k-1].keys())
        candidates = set()
        
        for i in range(len(prev_itemsets)):
            for j in range(i+1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    if self._has_frequent_subsets(union, k-1):
                        candidates.add(union)
        
        return candidates
    
    def _has_frequent_subsets(self, itemset, k):
        """Перевіряє, чи всі k-елементні підмножини є частими"""
        for subset in combinations(itemset, k):
            if frozenset(subset) not in self.frequent_itemsets[k]:
                return False
        return True
    
    def _count_support(self, itemset):
        """Підраховує підтримку для набору елементів"""
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count
    
    def get_frequent_itemsets(self, min_length=1):
        """Повертає всі часті набори елементів"""
        result = {}
        for k, itemsets in self.frequent_itemsets.items():
            if k >= min_length:
                result[k] = itemsets
        return result
    
    def get_support(self, itemset):
        """Повертає підтримку для конкретного набору"""
        itemset = frozenset(itemset) if not isinstance(itemset, frozenset) else itemset
        n_transactions = len(self.transactions)
        
        for k, itemsets in self.frequent_itemsets.items():
            if itemset in itemsets:
                return itemsets[itemset] / n_transactions
        
        count = self._count_support(itemset)
        return count / n_transactions