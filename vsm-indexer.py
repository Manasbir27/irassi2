import os
import math
from collections import defaultdict
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class VSMIndexer:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.dictionary = defaultdict(lambda: {"df": 0, "postings": []})
        self.doc_lengths = {}
        self.N = 0
        self.stemmer = PorterStemmer()
        self.doc_id_map = {}

    def build_index(self):
        doc_id = 1
        for filename in os.listdir(self.corpus_path):
            if filename.endswith(".txt"):
                self.N += 1
                self.doc_id_map[filename] = doc_id
                with open(os.path.join(self.corpus_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                self.process_document(doc_id, content)
                doc_id += 1

        self.calculate_doc_lengths()
        self.write_index_to_file()

    def process_document(self, doc_id, content):
        term_freq = defaultdict(int)
        tokens = word_tokenize(content.lower())
        for token in tokens:
            stem = self.stemmer.stem(token)
            term_freq[stem] += 1

        for term, tf in term_freq.items():
            if doc_id not in [posting[0] for posting in self.dictionary[term]["postings"]]:
                self.dictionary[term]["df"] += 1
            # Store log(tf) directly in postings
            self.dictionary[term]["postings"].append((doc_id, 1 + math.log10(tf) if tf > 0 else 0))

    def calculate_doc_lengths(self):
        for term, data in self.dictionary.items():
            for doc_id, log_tf in data["postings"]:
                self.doc_lengths[doc_id] = self.doc_lengths.get(doc_id, 0) + log_tf ** 2

        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])

    def write_index_to_file(self):
        with open("dictionary.json", "w") as f:
            json.dump(self.dictionary, f)

        with open("doc_lengths.json", "w") as f:
            json.dump(self.doc_lengths, f)

        with open("doc_count.txt", "w") as f:
            f.write(str(self.N))

        with open("doc_id_map.json", "w") as f:
            json.dump(self.doc_id_map, f)

if __name__ == "__main__":
    corpus_path = r"C:\Users\Asus\Desktop\assignment_2\Corpus"
    indexer = VSMIndexer(corpus_path)
    indexer.build_index()
    print(f"Indexing complete. Processed {indexer.N} documents.")