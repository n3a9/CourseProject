import emoji
import re
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack


train_df = pd.read_json("../../data/train.jsonl", lines=True)
test_df = pd.read_json("../../data/test.jsonl", lines=True)

"""
Preprocess string
- Remove users
- Replace emojis
"""


def preprocess(s):
    emoji.demojize(s)
    s = s.replace("@USER USER USER", "")
    s = s.replace("@USER USER", "")
    s = s.replace("@USER", "")
    return s


"""
Append all context and train in single Tfidf
"""
tf_idf_context = TfidfVectorizer(ngram_range=(1, 2))
X_train_context = tf_idf_context.fit_transform(
    train_df["context"].apply(lambda x: " ".join(x))
)


"""
Response Tfidf
"""
tf_idf_response = TfidfVectorizer(ngram_range=(1, 2))
X_train_response = tf_idf_response.fit_transform(train_df["response"])
X_train = hstack([X_train_context, X_train_response])

"""
Model
"""
svm = SGDClassifier(verbose=2)
svm.fit(X_train, train_df["label"])

"""
Test Data Tfidf
"""
X_test_context = tf_idf_context.transform(
    test_df["context"].apply(lambda x: " ".join(x))
)
X_test_response = tf_idf_response.transform(test_df["response"])
X_test = hstack([X_test_context, X_test_response])

"""
Predictions and write to answer.txt
"""
predictions = svm.predict(X_test)

f = open("predictions.txt", "w")
for p in predictions:
    f.write(str(p[1]) + "\n")
f.close()

# predictions = []

# f = open("predicitions.txt", "r")
# for i in test_df["id"]:
#     predictions.append(float(f.readline()))
# f.close()

f = open("answer.txt", "w")
for x, y in zip(test_df["id"], predictions):
    f.write(x + "," + y + "\n")
f.close()
