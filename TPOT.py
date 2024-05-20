import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from sklearn import metrics

train_data = pd.read_csv("MIMIC\\data_train_valid_4w.csv")
test_data = pd.read_csv("MIMIC\\data_test_4w.csv")


tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

label_list = np.array(test_data.iloc[:, -1])
prob_list = tpot.predict_proba(test_data.iloc[:, :-1])[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
print(f"AUROC on the test set is {metrics.auc(fpr, tpr)}")
