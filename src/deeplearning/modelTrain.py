from models import CNN_MODEL
from dataProcessor import DataProcessorBERT
import utils
import json
import pandas as pd
import math
import tensorflow as tf
from numpy.random import seed
import sys
seed(50)
tf.random.set_seed(7)

def performTraining(parameters_filename=None):
    
    # Get parameters
    print('-----Parameter Reading Started-----\n')
    if parameters_filename is None:
        f = open('./parameters/params_config.json')
    else:
        f = open(parameters_filename)
    
    parameters = json.load(f)
    print('-----Parameter Reading Complete-----\n')


    # Process data to generate features
    print('-----Data Processing Started-----\n')
    dataprocessor = DataProcessorBERT(parameters,train=True)
    dataprocessor.processData()
    dataprocessor.buildFeatures()
    print('-----Data Processing Complete-----\n')
    
    # Read features
    print('-----Reading Features Started-----\n')
    filename = parameters['features-path'] + parameters['features-train-filename']
    features_df = pd.read_json(filename)
    features_df['len_r'] = features_df['tokenized'].apply(lambda x: len(x))
 
    features_df = features_df.sample(frac = 1)
    features_df.sort_values(by=['len_r'])

    sorted_data = [(data[0], data[1]) for data in features_df.values]    
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_data, output_types=(tf.int32, tf.int32))
    BATCH_SIZE = parameters['batch_size']
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

    print('-----Reading Features Completed-----\n')
    
    # Split data into training/validation
    print('-----Data Split Started-----\n')
    if parameters['train_test_split'] <= 0:
        parameters['train_test_split'] = 0.1

        
    TOTAL_BATCHES = math.ceil(len(sorted_data) / BATCH_SIZE)
    TEST_BATCHES = int(TOTAL_BATCHES * parameters['train_test_split'])

    batched_dataset.shuffle(TOTAL_BATCHES)
        
    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)

    print('-----Data Split Complete-----\n')
    
    # Build Model
    print('-----Model Building Started-----\n')
    tokenizer = utils.get_BERT_Tokenizer()

    parameters['vocabulary_size'] = len(tokenizer.vocab)
    parameters['max_length'] = features_df['len_r'].max()
    
    model = CNN_MODEL(parameters)

    model.compile(loss="binary_crossentropy",
                   optimizer="adam",
                   metrics=utils.METRICS)
    
    model.fit(train_data,
              validation_data = test_data,
              epochs=parameters['n_epochs'],
              verbose=parameters['verbose'])
    
    print('\n-----Model Building Complete-----\n')

    # Save the trained model
    if parameters['trained-model-save'] == 'X':
        filename = parameters['trained-model-path'] + parameters['trained-model-weight-filename']
        model.save_weights(filename)

        print('-----Model Saved to disk--------')
         
if __name__ == '__main__':
    parameters_filename = None
    
    if len(sys.argv) == 2:
        parameters_filename = sys.argv[1]
    
    performTraining(parameters_filename)