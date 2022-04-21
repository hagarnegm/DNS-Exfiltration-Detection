import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode(longest_word, sld):
    le = LabelEncoder()
    le.classes_ = np.load(r"D:\Hagar\Documents\uOttawa\Second Term\ELG7186 -  AI for Cyber Security\3- Assignments\Assignment 2\assignment2-hagarnegm\models\encoded_sld.npy", allow_pickle=True)

    if sld in le.classes_:
        encoded_sld = le.transform([sld])[0]
    else:
        encoded_sld = -1
    return len(longest_word), int(encoded_sld)
