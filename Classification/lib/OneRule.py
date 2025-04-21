import pandas as pd
import numpy as np
from collections import defaultdict

class OneRuleClassifier:
    def __init__(self):
        self.rule = None

    def fit(self, X, y):
        best_feature = None
        best_rule = None
        best_error = float('inf')

        for feature in X.columns:
            rule, error = self._create_rule(X[feature], y)

            if error < best_error:
                best_feature = feature
                best_rule = rule
                best_error = error
        
        self.rule = (best_feature, best_rule)
    
    def _create_rule(self, feature_column, y):
        rule_dict = {}
        error_count = 0
        
        grouped = defaultdict(lambda: defaultdict(int))

        for feature_value, label in zip(feature_column, y):
            grouped[feature_value][label] += 1
        
        for feature_value, label_counts in grouped.items():
            most_common_label = max(label_counts, key=label_counts.get)
            rule_dict[feature_value] = most_common_label
            error_count += sum(label_counts.values()) - label_counts[most_common_label]
        
        return rule_dict, error_count
    
    def predict(self, X):
        if self.rule is None:
            raise ValueError("Model has not been trained yet.")
        
        feature, rule = self.rule

        return np.array(X[feature].map(lambda x: rule.get(x, np.random.choice(list(rule.values())))))
    
    def name(self):
        return "OneRule"