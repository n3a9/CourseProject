import os
import re
#import tensorflow as tf
from bert import bert_tokenization
import keras.backend as K
import keras.metrics
from keras.models import model_from_json
import nltk
nltk.download('words')
from nltk.corpus import words, brown

# Pre-trained BERT tokenizer
def get_BERT_Tokenizer():
    path = os.getcwd()[:os.getcwd().rfind('/')] + '/deeplearning/'
    vocab_file = path + 'uncased_L-12_H-768_A-12' + '/vocab.txt'
    tokenizer = bert_tokenization.FullTokenizer(vocab_file=vocab_file,
                                                do_lower_case=True)
    return tokenizer

def tokenize_text(text,tokenizer):
    text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    return text

# Custom metric for f1-score calculation adjusted from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1_score(y_true, y_pred):
    # Recall metric. Only computes a batch-wise average of recall,
    # a metric for multi-label classification of how many relevant items are selected.
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    # Precision metric. Only computes a batch-wise average of precision,
    # a metric for multi-label classification of how many selected items are relevant.
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision*recall) / (precision+recall))    

# Other metrics for validation
METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    f1_score
]

# Hash tag expanding helper functions

def longest_word(phrase, words):
    current_try = phrase
    while current_try:
        if current_try in words or current_try.lower() in words:
            return current_try
        current_try = current_try[:-1]
    # if nothing was found, return the original phrase
    return phrase

def partitioner(hashtag, words):
    while hashtag:
        word_found = longest_word(hashtag, words)
        yield word_found
        hashtag = hashtag[len(word_found):]

def partition_hashtag(text, words):
    return re.sub(r'#(\w+)', lambda m: ' '.join(partitioner(m.group(1), words)), text)

if __name__ == '__main__':
    print(get_BERT_Tokenizer().tokenize('this is tokenizer'))
    print(tokenize_text('this is tokenizer',get_BERT_Tokenizer()))