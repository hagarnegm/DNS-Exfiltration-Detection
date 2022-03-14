import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode(longest_word, sld):
    le = LabelEncoder()
    le.classes_ = np.load("./models/encoded_sld.npy", allow_pickle=True)

    if sld in le.classes_:
        encoded_sld = le.transform([sld])[0]
    else:
        encoded_sld = -1
    return len(longest_word), encoded_sld
