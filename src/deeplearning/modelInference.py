import sys
from dataProcessor import DataProcessorBERT
from models import CNN_MODEL
import keras
import utils
import pandas as pd
import json
import tensorflow as tf

def performInference(parameters_filename=None):
    
    # Get parameters
    print('-----Parameter Reading Started-----\n')
    if parameters_filename is None:
        f = open('./parameters/params_config.json')
    else:
        f = open(parameters_filename)

    parameters = json.load(f)
    print('-----Parameter Reading Complete-----\n')
    

    # Process test data to generate features
    print('-----Data Processing Started-----\n')
    dataprocessor = DataProcessorBERT(parameters,train=False)
    dataprocessor.processData()
    dataprocessor.buildFeatures()
    print('-----Data Processing Complete-----\n')
    
    # Read features
    filename = parameters['features-path'] + parameters['features-test-filename']
    features_df = pd.read_json(filename)
    features_df['len_r'] = features_df['tokenized'].apply(lambda x: len(x))

    features_df.sort_index(inplace=True)

    data_list = [(data[0]) for data in features_df.values]
  
    processed_dataset = tf.data.Dataset.from_generator(lambda: data_list, output_types=(tf.int32))
    

    BATCH_SIZE = parameters['batch_size']
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, )))

    # Load Model
    print('-----Model Loading Started-----\n')
    tokenizer = utils.get_BERT_Tokenizer()
    parameters['vocabulary_size'] = len(tokenizer.vocab)
    
    max_length_json = open('./parameters/maxlength.txt', 'r')
    parameters['max_length'] = int(max_length_json.read())
    
    model = CNN_MODEL(parameters)
    model.build((parameters['max_length'],parameters['embedding_dimensions']))

    filename = parameters['trained-model-path'] + parameters['trained-model-weight-filename']
    model.load_weights(filename)
    
    print('\n-----Model Loading Complete-----\n')

    # Perform Prediction
    print('-----Prediction Started-----\n')
    preds = model.predict(batched_dataset)
    #print(preds[:10])
    t=parameters['prediction-threshold']
    preds[preds>=t] = 1
    preds[preds<t] = 0
    #print(preds[:10])
    print('-----Prediction Complete-----\n')

    # Build Answers
    print('-----Building Answers Started-----\n')
    filename = parameters['data-path'] + parameters['test-data-filename']
    test_df = pd.read_json(path_or_buf= filename,lines = True)
    
    f = open(parameters["answer-file"], "w")
    for x, y in zip(test_df['id'], preds):
        if y==1:
            result = 'SARCASM'
        else:
            result = 'NOT_SARCASM'
        f.write(x + ',' + result +'\n')
    f.close()
    
    print('-----Building Answers Complete----\n')

if __name__ == '__main__':
    parameters_filename = None

    if len(sys.argv) == 2:
        parameters_filename = sys.argv[1]
    
    performInference(parameters_filename)

