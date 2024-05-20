import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
from sklearn import metrics

data_train_org = pd.read_csv("MIMIC\\data_train_4w.csv")
data_val_org = pd.read_csv("MIMIC\\data_validation_4w.csv")
data_test_org = pd.read_csv("MIMIC\\data_test_4w.csv")
data_train_val = pd.concat([data_train_org, data_val_org]).drop_duplicates().reset_index(drop=True)
data_train_val.to_csv("MIMIC\\data_train_valid_4w.csv", index=False)

train_data = TabularDataset("MIMIC\\data_train_valid_4w.csv")
train_data.head()

label = 'label'
print(f"Unique classes: {list(train_data[label].unique())}")
predictor = TabularPredictor(label=label).fit(train_data)
test_data = TabularDataset("MIMIC\\data_test_4w.csv")

label_list = np.array(test_data.iloc[:, -1])
prob_list = np.array(predictor.predict_proba(test_data)[1])
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
print(f"AUROC on the test set is {metrics.auc(fpr, tpr)}")