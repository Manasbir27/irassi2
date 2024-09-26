import os
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Preprocessing functions
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    """Tokenizes, removes stop words, punctuations, and stems the words."""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Step 1: Indexing the corpus
class Indexer:
    def __init__(self):
        self.dictionary = defaultdict(lambda: {'df': 0, 'postings': []})
        self.doc_lengths = {}  # Store document length (for normalization)
        self.N = 0  # Total number of documents

    def index_document(self, doc_name, content):
        """Indexes a single document by file name."""
        term_freq = defaultdict(int)
        tokens = preprocess(content)
        
        # Count term frequencies in the document
        for token in tokens:
            term_freq[token] += 1
        
        # Update the dictionary and document frequency
        for term, freq in term_freq.items():
            if len(self.dictionary[term]['postings']) == 0 or self.dictionary[term]['postings'][-1][0] != doc_name:
                self.dictionary[term]['df'] += 1
            self.dictionary[term]['postings'].append((doc_name, freq))
        
        # Compute document length for normalization
        self.doc_lengths[doc_name] = self.compute_doc_length(term_freq)
        self.N += 1

    def compute_doc_length(self, term_freq):
        """Computes the length of a document for normalization."""
        length = 0
        for freq in term_freq.values():
            length += (1 + math.log10(freq))**2
        return math.sqrt(length)

    def build_index(self, folder_path):
        """Indexes all documents in the folder."""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.index_document(filename, content)

