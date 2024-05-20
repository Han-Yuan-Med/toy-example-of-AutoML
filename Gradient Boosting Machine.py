import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics


train_data = pd.read_csv("MIMIC\\data_train_4w.csv")
val_data = pd.read_csv("MIMIC\\data_validation_4w.csv")
test_data = pd.read_csv("MIMIC\\data_test_4w.csv")

performance_list = []
for num in range(100, 1001, 100):
    gbm = HistGradientBoostingClassifier(max_iter=num)
    gbm.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    prob_list = gbm.predict(val_data.iloc[:, :-1])
    label_list = np.array(val_data.iloc[:, -1])
    performance_list.append(roc_auc_score(label_list, prob_list))

(np.argmax(performance_list)+1)*100

gbm = HistGradientBoostingClassifier(max_iter=600)
gbm.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])

label_list = np.array(test_data.iloc[:, -1])
prob_list = gbm.predict(test_data.iloc[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
print(f'AUROC results on the test set is {metrics.auc(fpr, tpr)}')
