import re
import sys
import utils
import json
import emoji
import demoji
import sys
import re

import nltk
nltk.download('words')
from nltk.corpus import words, brown

class Preprocessor(object):
    """
    The purpose of this class is to preprocess the incoming text data so that 
    it can be used later by other modules
    """
    def __init__(self):
        """
        Constructor initializing the attributes
        """
        demoji.download_codes()
    
    def expand_contractions(self,text):
        """
        The purpose of this method is to expand any contractions that are present
        in the incoming text
        """
        # Read contractions and create dictionary
        with open('../common/contractions.txt') as f: 
            contractions = f.read()
        contractions_dict = json.loads(contractions)
        
        # Regular expression for finding contractions
        contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

        # Replace contractions in text
        def replace(match):
            return contractions_dict[match.group(0)]

        text = contractions_re.sub(replace, text)
        return text

    def replace_emoji(self,text):
        """
        The purpose of this method is to replace emojis in text with
        their respective description
        """
        text = emoji.demojize(self.text)
        return text

    def remove_emoji(self,text):
        """
        The purpose of this method is to remove emojis in text
        """
        text = str.strip(demoji.replace(text))
        return text

    def remove_user(self,text):
        """
        The purpose of this method is to remove @USER USER USER,@USER USER
        and @USER in text
        """
        text = text.replace('@USER USER USER', '')
        text = text.replace('@USER USER', '')
        text = text.replace('@USER', '')
        return text

    def remove_numbers(self,text):
        """
        The purpose of this method is to remove numbers from text
        """
        text = re.sub('[0-9]','',text)
        return text
    
    def remove_hashtag(self,text):
        """
        The purpose of this method is to remove hashtags
        """
        text = re.sub(r"#(\w+)",' ', text)
        return text

    def expand_hashtags(self,text):
        word_dictionary = list(set(words.words()))
        hashtags = re.findall(r"#(\w+)", text )
        for hashtag in hashtags:
            hashtag = '#' + hashtag
            split = re.sub(r'\W*\b\w{1}\b', '',utils.partition_hashtag(hashtag,word_dictionary))
            text = text.replace(hashtag,split)
        return text

    def remove_alphanumerics(self,text):
        """
        The purpose of this method is to remove any non-alphabets
        """
        text = re.sub('[^a-zA-Z]', ' ', text)
        return text

    def remove_2_letter_words(self, text):
        """
        The purpose of this method is to remove 'n' letter words from text
        """
        text = re.sub(r'\W*\b\w{1,2}\b', '',text )
        return text

    def remove_white_spaces(self,text):
        """
        The purpose of this method is to remove any whitespces from text
        """
        text = ' '.join(text.split())
        return text

    def process_text_bert(self,text):
        """
        The purpose of this method is to process the text for 
        deep learning models
        """
        text = self.remove_user(text)
        text = self.remove_emoji(text)
        #text = self.expand_hashtags(text)
        text = str.lower(text)
        #text = self.expand_contractions(text)
        #text = self.remove_hashtag(text)
        #text = self.remove_alphanumerics(text)
        text = self.remove_numbers(text)
        text = self.remove_white_spaces(text)
        return text


if __name__ == '__main__':

    text = "@USER Doesn ’ t @USER USER USER Test 🤣 😂 🤣 OF what's @USER USER contraction 29"
    processor = Preprocessor()
    print(processor.process_text_dl(text))


    