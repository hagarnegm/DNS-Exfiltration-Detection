import json
import pickle
from datetime import datetime

from src.utils import encode
from src.features.build_features import extract_features

import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from kafka import KafkaProducer


# This function loads the weights of the trained random forest model and predicts the class of raw DNS queries
# using features extracted from the query
def detect_dns_exf(features):
    """
    :param features: Features extracted from the DNS query
    :return: The predicted label and confidence score of the class of DNS query (benign or malicious)
    """
    rf = pickle.load(open("./models/rf_model", 'rb'))
    features = np.array(features).reshape(1, -1)
    prediction_label = rf.predict(features)[0]
    score = rf.predict_proba(features)[0]
    return prediction_label, score[score.argmax()]


# Connecting the Kafka Consumer to the input topic "ml-raw-dns" in order to consumer the incoming raw dns queries
# starting from the beginning of the queue
consumer = KafkaConsumer('ml-raw-dns', bootstrap_servers=['localhost:9092'], auto_offset_reset='earliest',
                         consumer_timeout_ms=5000)
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: json.dumps(x).encode('utf-8'))

predictions = []
# For each query, features are extracted and passed through a model to predict whether the dns query
# is benign or malicious
for query in consumer:
    domain = query.value.decode('utf-8')

    # Extract features from domain
    features = extract_features(domain)

    # Encode categorical features
    longest_word_length, encoded_sld = encode(features[-4], features[-3])
    encoded_features = features
    encoded_features[-4] = longest_word_length
    encoded_features[-3] = encoded_sld

    prediction_label, score = detect_dns_exf(encoded_features)

    pred = features
    pred.insert(0, domain)

    pred.append(int(prediction_label))
    pred.append(float(score))

    predictions.append(pred)

    if pred[-2] == 1:
        print("WARNING: MALICIOUS DNS QUERY")
        producer.send('ml-dns-predictions', value=pred)

    print("{0} - DNS Query: {1}, Prediction Label: {2}"
          .format(datetime.now(), domain, prediction_label))


KafkaConsumer.close()


# Predictions are then added to a dataframe and saved to a csv file along with the extracted features
columns = ['domain', 'FQDN_count', 'subdomain_length', 'upper', 'lower', 'numeric', 'entropy', 'special', 'labels',
           'labels_max', 'labels_average', 'longest_word', 'sld', 'length', 'subdomain', 'predicted_label', 'score']
predictions = pd.DataFrame(predictions, columns=columns)
predictions.to_csv("../data/predictions.csv",  index=False)
