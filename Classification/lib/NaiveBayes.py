import numpy as np
import pandas as pd
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, alpha=0.1):
        self.priors = {}
        self.likelihoods = {}
        self.likelihoods_unseen = {}
        self.classes = []
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.likelihoods = {c: defaultdict(lambda: defaultdict(float)) for c in self.classes}
        self.likelihoods_unseen = {c: defaultdict(float) for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]

            for feature in X.columns:
                feature_counts = X_c[feature].value_counts().to_dict()
                total_count = len(X_c)
                unique_values = X[feature].nunique()

                for value in feature_counts.keys():
                    if len(feature_counts) != unique_values:
                        self.likelihoods[c][feature][value] = (feature_counts[value] + self.alpha) / (total_count + self.alpha * unique_values)
                    else:
                        self.likelihoods[c][feature][value] = feature_counts[value] / total_count

                self.likelihoods_unseen[c][feature] = self.alpha / (total_count + self.alpha * unique_values)

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_probs = {}

            for c in self.classes:
                prob = self.priors[c]

                for feature, value in row.items():
                    prob *= self.likelihoods[c][feature].get(value, self.likelihoods_unseen[c][feature])
                class_probs[c] = prob

            predictions.append(max(class_probs, key=class_probs.get))

        return np.array(predictions)

    def name(self):
        return "NaiveBayes"