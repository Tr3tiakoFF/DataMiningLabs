from collections import Counter


class FPNode:
    """Вузол FP-дерева"""
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None
    
    def increment(self, count=1):
        """Збільшує лічильник вузла"""
        self.count += count
    
    def display(self, level=0):
        """Відображає дерево (для налагодження)"""
        print('  ' * level + f'{self.item}: {self.count}')
        for child in self.children.values():
            child.display(level + 1)


class FPTree:
    """FP-дерево"""
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.frequent_items = {}
        self.header_table = {}
        self.root = FPNode()
        
        self._build_tree(transactions)
    
    def _build_tree(self, transactions):
        """Будує FP-дерево"""
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        self.frequent_items = {
            item: count for item, count in item_counts.items()
            if count >= self.min_support
        }
        
        if not self.frequent_items:
            return
        
        sorted_items = sorted(self.frequent_items.items(), 
                            key=lambda x: x[1], reverse=True)
        
        for item, _ in sorted_items:
            self.header_table[item] = None
        
        for transaction in transactions:
            filtered_items = [item for item in transaction 
                            if item in self.frequent_items]
            filtered_items.sort(key=lambda x: self.frequent_items[x], reverse=True)
            
            if filtered_items:
                self._insert_transaction(filtered_items, self.root)
    
    def _insert_transaction(self, transaction, node):
        """Вставляє транзакцію в дерево"""
        if not transaction:
            return
        
        first_item = transaction[0]
        
        if first_item in node.children:
            child = node.children[first_item]
            child.increment()
        else:
            child = FPNode(first_item, 1, node)
            node.children[first_item] = child
            
            self._update_header_table(first_item, child)
        
        if len(transaction) > 1:
            self._insert_transaction(transaction[1:], child)
    
    def _update_header_table(self, item, node):
        """Оновлює таблицю заголовків"""
        if self.header_table[item] is None:
            self.header_table[item] = node
        else:
            current = self.header_table[item]
            while current.node_link is not None:
                current = current.node_link
            current.node_link = node
    
    def get_prefix_paths(self, item):
        """Отримує префіксні шляхи для елемента"""
        paths = []
        node = self.header_table.get(item)
        
        while node is not None:
            if node.parent.item is not None:
                path = []
                count = node.count
                current = node.parent
                
                while current.item is not None:
                    path.append(current.item)
                    current = current.parent
                
                if path:
                    paths.append((path[::-1], count))
            
            node = node.node_link
        
        return paths


class FPGrowthAlgorithm:
    """Алгоритм FP-Growth"""
    def __init__(self, min_support=0.2):
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.transactions = []
    
    def fit(self, transactions):
        """
        Знаходить всі часті набори елементів за допомогою FP-Growth
        
        Args:
            transactions: список транзакцій
        """
        self.transactions = transactions
        n_transactions = len(transactions)
        min_support_count = int(self.min_support * n_transactions)
        
        fp_tree = FPTree(transactions, min_support_count)
        
        self.frequent_itemsets = {}
        self._fp_growth(fp_tree, [], n_transactions)
        
        return self.frequent_itemsets
    
    def _fp_growth(self, fp_tree, alpha, total_transactions):
        """Рекурсивна функція FP-Growth"""
        items = sorted(fp_tree.frequent_items.items(), key=lambda x: x[1])
        
        for item, count in items:
            beta = alpha + [item]
            support = count / total_transactions
            
            key = len(beta)
            if key not in self.frequent_itemsets:
                self.frequent_itemsets[key] = {}
            self.frequent_itemsets[key][frozenset(beta)] = count
            
            prefix_paths = fp_tree.get_prefix_paths(item)
            
            if prefix_paths:
                conditional_transactions = []
                for path, path_count in prefix_paths:
                    for _ in range(path_count):
                        conditional_transactions.append(path)
                
                conditional_tree = FPTree(conditional_transactions, 
                                        int(self.min_support * total_transactions))
                
                if conditional_tree.frequent_items:
                    self._fp_growth(conditional_tree, beta, total_transactions)
    
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
        
        return 0.0