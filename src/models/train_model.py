import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load training dataset
dns_exf_feats = pd.read_csv("../data/training_dataset.csv")

# Data Preprocessing
# Drop timestamp and drop any row that contains na values
dns_exf_feats.drop(columns=['timestamp'], inplace=True)
dns_exf_feats.dropna(inplace=True)

# Replace longest word column values with the length of the words instead of the actual words
dns_exf_feats.longest_word = dns_exf_feats.longest_word.apply(lambda x: len(x))

# Encode sld column, since Random Forest models in sklearn do not support categorical features
le = LabelEncoder()
le.fit(dns_exf_feats.sld)
dns_exf_feats.sld = le.transform(dns_exf_feats.sld)

# Saving encoder classes for later use when extracting features from raw dns queries
np.save('../../models/encoded_sld.npy', le.classes_)

# Plotting frequency of classes to check if data is imbalanced
labels = dns_exf_feats.Label.value_counts().index
frequencies = dns_exf_feats.Label.value_counts()
class_freq = sns.barplot(x=labels, y=frequencies)
class_freq.set(xlabel='Class', ylabel='Frequency')
plt.show()
fig = class_freq.get_figure()
fig.savefig("../../reports/figures/class_frequencies.png")

# Splitting data into train set and test set (70/30)
X, y = dns_exf_feats.iloc[:, 0:14], dns_exf_feats.iloc[:, 14]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fitting a Random Forest classifier using 50 trees on data
rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.tight_layout()
plt.savefig('../../reports/figures/confusion_matrix.png', bbox_inches='tight')
plt.show()

# Saving model weights
model = open("../../models/rf_model", 'wb')
pickle.dump(rf, model)
model.close()
