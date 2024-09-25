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

# Step 2: Ranked Retrieval (searching with cosine similarity)
class Searcher:
    def __init__(self, indexer):
        self.indexer = indexer

    def search(self, query):
        """Searches for the top 10 relevant documents based on the query."""
        query_tokens = preprocess(query)
        query_vector, query_length = self.compute_query_vector(query_tokens)
        scores = defaultdict(float)

        # Calculate cosine similarity for each document
        for term, query_weight in query_vector.items():
            if term in self.indexer.dictionary:
                postings = self.indexer.dictionary[term]['postings']
                idf = math.log10(self.indexer.N / self.indexer.dictionary[term]['df'])
                for doc_name, tf in postings:
                    tf_weight = 1 + math.log10(tf)
                    scores[doc_name] += tf_weight * query_weight * idf

        # Normalize scores by the magnitudes of the document and query vectors
        for doc_name in scores:
            if self.indexer.doc_lengths[doc_name] > 0:  # Prevent division by zero
                scores[doc_name] /= (self.indexer.doc_lengths[doc_name] * query_length)  # Cosine similarity formula

        # Sort by score and return top 10 results
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked_docs[:10]

    def compute_query_vector(self, query_tokens):
        """Computes the tf-idf vector for the query and returns its magnitude."""
        term_freq = defaultdict(int)
        for token in query_tokens:
            term_freq[token] += 1

        query_vector = {}
        query_length = 0  # Magnitude of the query vector

        # Calculate TF-IDF and query length
        for term, freq in term_freq.items():
            tf_weight = 1 + math.log10(freq)
            idf_weight = math.log10(self.indexer.N / (self.indexer.dictionary[term]['df'] if term in self.indexer.dictionary else 1))
            query_vector[term] = tf_weight * idf_weight
            query_length += (query_vector[term])**2

        query_length = math.sqrt(query_length)  # Compute the magnitude of the query vector
        return query_vector, query_length

# Main driver function
def main():
    folder_path = r"C:\Users\Asus\Desktop\irassi2\Corpus"  # Hardcoded path to the corpus folder
    indexer = Indexer()
    indexer.build_index(folder_path)
    searcher = Searcher(indexer)

    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        results = searcher.search(query)

        print("\nTop 10 relevant documents:")
        if results:
            for doc_name, score in results:
                print(f"{doc_name}: Score {score:.10f}")
        else:
            print("No relevant documents found.")

# To run the search system, uncomment the line below
if __name__ == "__main__":
    main()
