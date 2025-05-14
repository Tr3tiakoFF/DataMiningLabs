from itertools import combinations


class AssociationRule:
    """Клас для представлення асоціативного правила"""
    def __init__(self, antecedent, consequent, support, confidence, lift=None):
        self.antecedent = frozenset(antecedent)
        self.consequent = frozenset(consequent)
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self):
        antecedent_str = ', '.join(sorted(self.antecedent))
        consequent_str = ', '.join(sorted(self.consequent))
        return f"{{{antecedent_str}}} => {{{consequent_str}}}"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        """Конвертує правило в словник"""
        return {
            'antecedent': list(self.antecedent),
            'consequent': list(self.consequent),
            'support': self.support,
            'confidence': self.confidence,
            'lift': self.lift
        }


class RulesGenerator:
    """Генератор асоціативних правил"""
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence
        self.rules = []
    
    def generate_rules(self, algorithm_instance, min_confidence=None):
        """
        Генерує асоціативні правила з частих наборів
        
        Args:
            algorithm_instance: екземпляр AprioriAlgorithm або FPGrowthAlgorithm
            min_confidence: мінімальна довіра (за замовчуванням використовується self.min_confidence)
        
        Returns:
            список асоціативних правил
        """
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        self.rules = []
        frequent_itemsets = algorithm_instance.get_frequent_itemsets(min_length=2)
        
        for length, itemsets in frequent_itemsets.items():
            for itemset in itemsets:
                if len(itemset) >= 2:
                    self._generate_rules_for_itemset(itemset, algorithm_instance, min_confidence)
        
        self.rules.sort(key=lambda x: x.confidence, reverse=True)
        
        return self.rules
    
    def _generate_rules_for_itemset(self, itemset, algorithm_instance, min_confidence):
        """Генерує всі можливі правила для одного частого набору"""
        items = list(itemset)
        itemset_support = algorithm_instance.get_support(itemset)
        
        for i in range(1, len(items)):
            for antecedent in combinations(items, i):
                consequent = [item for item in items if item not in antecedent]
                
                if consequent:
                    antecedent_set = frozenset(antecedent)
                    consequent_set = frozenset(consequent)
                    
                    antecedent_support = algorithm_instance.get_support(antecedent_set)
                    
                    if antecedent_support > 0:
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= min_confidence:
                            consequent_support = algorithm_instance.get_support(consequent_set)
                            lift = confidence / consequent_support if consequent_support > 0 else float('inf')
                            
                            rule = AssociationRule(
                                antecedent=antecedent,
                                consequent=consequent,
                                support=itemset_support,
                                confidence=confidence,
                                lift=lift
                            )
                            self.rules.append(rule)
    
    def filter_rules(self, min_support=None, min_confidence=None, min_lift=None):
        """
        Фільтрує правила за заданими критеріями
        
        Args:
            min_support: мінімальна підтримка
            min_confidence: мінімальна довіра
            min_lift: мінімальний lift
        
        Returns:
            відфільтровані правила
        """
        filtered_rules = self.rules.copy()
        
        if min_support is not None:
            filtered_rules = [rule for rule in filtered_rules if rule.support >= min_support]
        
        if min_confidence is not None:
            filtered_rules = [rule for rule in filtered_rules if rule.confidence >= min_confidence]
        
        if min_lift is not None:
            filtered_rules = [rule for rule in filtered_rules if rule.lift >= min_lift]
        
        return filtered_rules
    
    def get_rules_containing_item(self, item, in_antecedent=True, in_consequent=True):
        """
        Повертає правила, що містять певний елемент
        
        Args:
            item: елемент для пошуку
            in_antecedent: шукати в умові
            in_consequent: шукати в наслідку
        
        Returns:
            список правил
        """
        result = []
        
        for rule in self.rules:
            found = False
            
            if in_antecedent and item in rule.antecedent:
                found = True
            
            if in_consequent and item in rule.consequent:
                found = True
            
            if found:
                result.append(rule)
        
        return result
    
    def print_rules(self, rules=None, top_n=None):
        """
        Виводить правила у зручному форматі
        
        Args:
            rules: список правил (за замовчуванням всі правила)
            top_n: кількість топ-правил для виводу
        """
        if rules is None:
            rules = self.rules
        
        if top_n is not None:
            rules = rules[:top_n]
        
        print(f"{'Rule':<40} {'Support':<10} {'Confidence':<12} {'Lift':<8}")
        print("-" * 75)
        
        for rule in rules:
            print(f"{str(rule):<40} {rule.support:<10.3f} {rule.confidence:<12.3f} {rule.lift:<8.3f}")
    
    def export_rules_to_dict(self, rules=None):
        """
        Експортує правила у формат словника
        
        Args:
            rules: список правил (за замовчуванням всі правила)
        
        Returns:
            список словників з правилами
        """
        if rules is None:
            rules = self.rules
        
        return [rule.to_dict() for rule in rules]