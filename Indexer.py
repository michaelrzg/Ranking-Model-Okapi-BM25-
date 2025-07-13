import os
import code
from collections import defaultdict, Counter
import pickle
import math
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
        self.dataset = {}
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))
        self.idx2tok = dict()
        self.lookup_table = {}
        # load config data
        self.data_directory = cfg['data_dir']
        self.index_file = cfg["idx_file"]
        # load data from data file (so avglen can be generated)
        self.load_corups(self.data_directory)
        if self.index_file in os.listdir():
            try: 
                with open(self.index_file,"rb") as file:
                    self.inverted_index = pickle.load(file)
                print("index loaded")

            except Exception as e:
                print(e)
        else:
            print("index not found, recreating")

            # generate the inverted index from data
            self.generate_inverted_index()
            # save 
            self.save()
        self.avglen = self.lookup_table['avglen']
    def load_corups(self,dir):
        current_id = 0
        total_length = 0
        try:     
            files = os.listdir(dir) # open data file
            N = len(files)
            for file in files:
                path = os.path.join(dir,file) # get path to each file
                with open(path) as current:
                    full_text = current.read()
                    text_section = self.get_text(full_text)# get data undrer TExt: header
                    page = self.text_preprocessing(text_section)
                    total_length+=len(page)
                    self.dataset[current_id] = page
                    self.lookup_table[current_id] = {"title": file,"url":self.get_url(full_text),"text" : text_section}
                    current_id+=1
                    self.add_to_vocabulary(page)
            self.lookup_table['avglen'] =total_length/N
            self.lookup_table['N'] = N
        except Exception as e:
            print(e)


    def get_text(self,full_file):
        start_index = full_file.find("Text: ") # find the Text header
        if start_index != -1:
            text_content_start = start_index + len("Text: ") 
            extracted_text = full_file[text_content_start:].strip() # grab the rest of the file after the header
            return extracted_text
    def get_url(self,full_file):
        line = full_file.splitlines()[0].split(": ")[1]
        return line
    
    def text_preprocessing(self, text):
        text = text.lower() # lowercase
        tokens = word_tokenize(text) #tokenize
        cleaned_tokens = [re.sub(r'[^a-z0-9]', '', token) for token in tokens]
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

        for doc_id, text_content in self.dataset.items():
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

    def save(self):
        with open(self.index_file,"wb") as file:
            pickle.dump(self.inverted_index,file)

class SearchAgent:
    def __init__(self, indexer, cfg):
        self.indexer = indexer
        self.k1 = cfg['k1']
        self.b = cfg['b']
        self.N = indexer.lookup_table['N']
        self.inv = indexer.inverted_index

    def query(self, q_str):
        query = self.indexer.text_preprocessing(q_str) # preprocess query text
        scores = []
        for docID,text in self.indexer.dataset.items():
            bm25 = 0
            d = len(text)
            for t in query:
                if t not in self.inv:
                    continue 
                document_freq = len(self.inv[t])
                term_freq_in_doc = 0
                for existing_doc_id, freq_in_doc in self.inv[t]:
                    if existing_doc_id == docID:
                        term_freq_in_doc = freq_in_doc 
                        break
                term_freq = term_freq_in_doc

                if document_freq == 0: 
                    firsthalf = 0
                else:
                    firsthalf =( self.N - document_freq + 0.5)/(document_freq+0.5)
                    firsthalf = math.log(firsthalf,2)

                second_half = (term_freq * (self.k1 +1))/ (term_freq+self.k1 * (1 - self.b + (self.b * (d/self.indexer.avglen))))

                bm25 += firsthalf * second_half
            if bm25 > 0:
                scores.append((docID,bm25))
        scores.sort(key=lambda x: x[1], reverse=True)
        self.display_results(scores[:5])
        return scores


    def display_results(self, scores):
        for index,x in enumerate(scores):
            data = self.indexer.lookup_table[x[0]]
            print("Rank: ", index+1)
            print("DocID: ",x[0])
            print("Score: ", x[1])
            print('url: ', data['url'])
            print('filename: ', data['title'])
            print("\n\n")
            

CFG = {
    'idx_file': 'irdataset.idx',
    'data_dir': './data',
    'k1': 1.2,
    'b': 0.75,
}

if __name__ == "__main__":
    i = Indexer(CFG)
    q = SearchAgent(i, CFG)
    code.interact(local=dict(globals(), **locals()))
