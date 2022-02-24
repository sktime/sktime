# -*- coding: utf-8 -*-
"""Tests to check compliance with Model selection and evaluation in scikit."""

from classification.hybrid import HIVECOTEV2
from datasets import load_unit_test
from sklearn.model_selection import cross_val_score, train_test_split

##Cross validation

X, y = load_unit_test()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
hc2 = HIVECOTEV2(time_limit_in_minutes=1)
scores = cross_val_score(hc2, X, y, cv=5)
print(scores)
