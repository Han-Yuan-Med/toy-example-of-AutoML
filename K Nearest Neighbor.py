import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics


train_data = pd.read_csv("MIMIC\\data_train_4w.csv")
val_data = pd.read_csv("MIMIC\\data_validation_4w.csv")
test_data = pd.read_csv("MIMIC\\data_test_4w.csv")

performance_list = []
for neigh_num in range(2, 10, 1):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    prob_list = neigh.predict(val_data.iloc[:, :-1])
    label_list = np.array(val_data.iloc[:, -1])
    performance_list.append(roc_auc_score(label_list, prob_list))

(np.argmax(performance_list)+2

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

label_list = np.array(test_data.iloc[:, -1])
prob_list = neigh.predict(test_data.iloc[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
print(f'AUROC results on the test set is {metrics.auc(fpr, tpr)}')