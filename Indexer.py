import os
import configparser
import code
from collections import defaultdict, Counter
import pickle
import math
import operator

from tqdm import tqdm
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

class Indexer:

    def __init__(self, cfg):

        #download nltk stuff
        nltk.download('stopwords')
        nltk.download('wordnet')
        # setup
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.book = {}
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))
        self.idx2tok = dict()
        # load config data
        self.data_directory = cfg['data_dir']
        self.index_file = cfg["idx_file"]
        # load data from data file
        self.load_corups(self.data_directory)
        # generate the inverted index from data
        self.generate_inverted_index()
        # save 
        self.save_inv_index()
    
    def load_corups(self,dir):
        current_id = 0
        try:     
            files = os.listdir(dir) # open data file
            for file in files:
                path = os.path.join(dir,file) # get path to each file
                with open(path) as current:
                    full_text = current.read()
                    full_text = self.get_text(full_text)# get data undrer TExt: header
                    page = self.text_preprocessing(full_text)
                    self.book[current_id] = page
                    current_id+=1
                    self.add_to_vocabulary(page)
                    

        except Exception as e:
            print(e)
        print(len(self.tok2idx))

    def get_text(self,full_file):
        start_index = full_file.find("Text:") # find the Text header
        stop_index = full_file.find("Next:")
        if start_index != -1:
            text_content_start = start_index + len("Text: ") 
            extracted_text = full_file[text_content_start:stop_index].strip() # grab the rest of the file after the header
            return extracted_text

    def text_preprocessing(self, text):
        text = text.lower() # lowercase
        tokens = word_tokenize(text) #tokenize
        cleaned_tokens = [re.sub(r'[^a-z\s]', '', token) for token in tokens] # remove all non letters
        cleaned_tokens = [token for token in cleaned_tokens if token and len(token)>1] # removes empty strings created by prev line
        filtered_tokens = [word for word in cleaned_tokens if word not in self.stop_words] # remove stopwords  
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens] # finally lemmetization 
        return lemmatized_tokens

    def add_to_vocabulary(self,tokens): # from assignment page
        for token in tokens:
            if token not in self.tok2idx:
                # Assign a new index to the token
                self.idx2tok[self.tok2idx[token]] = token
    
    def generate_inverted_index(self):
        self.inverted_index = {}

        for doc_id, text_content in self.book.items():
            doc_word_counts = {} # for word frequencies in the curr page
            for word in text_content:
                doc_word_counts[word] = doc_word_counts.get(word, 0) + 1

            for word, freq in doc_word_counts.items():
                if word not in self.inverted_index:
                    self.inverted_index[word] = [(doc_id, freq)] # if word does not exist, create it
                else:
                    found_doc = False # if the word alr exists, find if the documnt is alr in the inv index
                    for i, (existing_doc_id, existing_freq) in enumerate(self.inverted_index[word]):
                        if existing_doc_id == doc_id: # if ti exists alr, update freqency
                            self.inverted_index[word][i] = (doc_id, existing_freq + freq)
                            found_doc = True
                            break
                    if not found_doc:
                        self.inverted_index[word].append((doc_id, freq)) # else add it as a new document frequency paor

    def save_inv_index(self):
        with open(self.index_file,"wb") as file:
            pickle.dump(self.inverted_index,file)



class SearchAgent:
    def __init__(self, indexer, cfg):
        raise NotImplementedError("Delete this lien and write your code here.")

    def query(self, q_str):
        raise NotImplementedError("Delete this lien and write your code here.")

    def display_results(self, results):
        raise NotImplementedError("Delete this lien and write your code here.")


CFG = {
    'idx_file': 'irbook.idx',
    'data_dir': './data',
    'k1': 1.2,
    'b': 0.75,
}

if __name__ == "__main__":
    i = Indexer(CFG)
    #q = SearchAgent(i, CFG)
    code.interact(local=dict(globals(), **locals()))
