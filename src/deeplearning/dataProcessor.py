import sys
sys.path.append('../common')
import os
from abc import ABC, abstractmethod
from preprocessor import Preprocessor
import pandas as pd
import utils
import json

# This is the abstract class for processing data
class AbstractDataProcessor(ABC):
    
    def __init__(self,parameters,train=True):
        self.parameters = parameters
        self.n_last_context = parameters['n_last_context']
        self.train = train
    
    def processData(self):
        pass

    def buildFeatures(self):
        pass

# This is the concrete class for CNN-BERT Model
class DataProcessorBERT(AbstractDataProcessor):

    def processData(self):
        """
        The purpose of this method is to process both train/test raw data
        """
        # Load the preprocessor
        preprocessor = Preprocessor()
        
        if self.train:
            filename = self.parameters['data-path'] + self.parameters['train-data-filename']
        else:
            filename = self.parameters['data-path'] + self.parameters['test-data-filename']

        # read the required file
        data_df = pd.read_json(path_or_buf= filename,lines = True)

        # concatenate response and last 'n' contexts together
        data_df['CONTEXT'] = data_df['context'].apply(lambda x: ' '.join(x[-self.n_last_context:]))
        data_df['text'] = data_df['CONTEXT'] + ' ' + data_df['response']
        data_df['text'] = data_df['text'].apply(lambda x: preprocessor.process_text_bert(x))

        # save the processed data
        if self.train:
            filename = self.parameters['processed-data-path'] + self.parameters['processed-train-data-filename']
            data_df[['text','label']].to_csv(filename)
        else:
            filename = self.parameters['processed-data-path'] + self.parameters['processed-test-data-filename']
            data_df[['text']].to_csv(filename)
        return

    def buildFeatures(self):
        """
        The purpose of this method is to build features using 
        the pre-trained BERT tokenizer
        """
        if self.train:
            filename = self.parameters['processed-data-path'] + self.parameters['processed-train-data-filename']
        else:
            filename = self.parameters['processed-data-path'] + self.parameters['processed-test-data-filename']

        data_df = pd.read_csv(filename)

        tokenizer = utils.get_BERT_Tokenizer()        
        data_df['tokenized'] = data_df['text'].apply(lambda x: utils.tokenize_text(x,tokenizer))
        
        if self.train:
            max_length = data_df['tokenized'].apply(len).max()
            
            max_length_json = {"max_length":max_length}
            with open('./parameters/maxlength.txt', "w") as json_file:
                json_file.write(str(max_length))
        else:
            max_length_json = open('./parameters/maxlength.txt', 'r')
            max_length = int(max_length_json.read())

        features_df = pd.DataFrame()
        features_df['tokenized'] = data_df['tokenized'].copy()
        if self.train:
            filename = self.parameters['features-path'] + self.parameters['features-train-filename']
            
            features_df['label'] = 0
            indexes = data_df[data_df['label']=='SARCASM'].index
            features_df.iloc[indexes,-1] = 1

            features_df.to_json(filename)
        else:
            filename = self.parameters['features-path'] + self.parameters['features-test-filename']
            features_df.to_json(filename)
        return
        

if __name__ == '__main__': 
    
    f = open('./parameters/params_config.json')
    parameters = json.load(f)
    
    dataprocessor = DataProcessorBERT(parameters,train=True)
    dataprocessor.processData()
    dataprocessor.buildFeatures()

    dataprocessor = DataProcessorBERT(parameters,train=False)
    dataprocessor.processData()
    dataprocessor.buildFeatures()
