#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import sys

# Load the dataset (replace 'path_to_file' with the actual path)
data = pd.read_csv('Creditcard_data.csv')

# Assuming 'Class' is the target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Define the models
models = [
    RandomForestClassifier(),
    SVC(),
    LogisticRegression(),
    XGBClassifier(),
    DecisionTreeClassifier()
]

# Redirect print output to a CSV file
output_file = 'model_results.csv'
original_stdout = sys.stdout
with open(output_file, 'w') as f:
    sys.stdout = f
    # Apply models to each sampling technique
    for model in models:
        print(f"Model: {model.__class__.__name__}")
        for sampling_method in ['simple', 'systematic', 'stratified', 'bootstrap', 'cross-validation']:
            if sampling_method == 'simple':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            elif sampling_method == 'systematic':
                sample_size = int(len(data) * 0.8)
                step = len(data) // sample_size
                indices = range(0, len(data), step)
                train_indices = list(indices)
                test_indices = [i for i in range(len(data)) if i not in train_indices]
                if len(train_indices) > 0 and len(test_indices) > 0:
                    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
                else:
                    print("Empty dataset encountered during systematic sampling.")
                    continue
            elif sampling_method == 'stratified':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            elif sampling_method == 'bootstrap':
                X_train, y_train = resample(X, y, replace=True, n_samples=int(0.8*len(data)), random_state=42)
                X_test, y_test = X[~X.index.isin(X_train.index)], y[~y.index.isin(y_train.index)]
            else:  # Cross-validation
                if len(X) > 0 and len(y) > 0:
                    model_cv = cross_val_score(model, X, y, cv=5)
                    print(f"Cross-Validation Scores: {model_cv}")
                    print(f"Mean CV Score: {np.mean(model_cv)}")
                else:
                    print("Empty dataset encountered during cross-validation.")
                continue  # Skip further processing for cross-validation
            
            # Fit the model
            if len(X_train) > 0 and len(y_train) > 0:
                model.fit(X_train, y_train)
            else:
                print("Empty training dataset.")
                continue
            
            # Evaluate the model
            if len(X_test) > 0 and len(y_test) > 0:
                accuracy = model.score(X_test, y_test)
                print(f"Sampling Method: {sampling_method.capitalize()}, Accuracy: {accuracy}")
            else:
                print("Empty testing dataset.")

        print("\n")

# Reset the stdout to the original
sys.stdout = original_stdout
print(f"Results saved to {output_file}")

