import json
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class VSMSearcher:
    def __init__(self):
        self.dictionary = {}
        self.doc_lengths = {}
        self.N = 0
        self.stemmer = PorterStemmer()
        self.doc_id_map = {}
        self.id_to_filename = {}
        self.load_index()

    def load_index(self):
        try:
            with open("dictionary.json", "r") as f:
                self.dictionary = json.load(f)
            with open("doc_lengths.json", "r") as f:
                self.doc_lengths = json.load(f)
            with open("doc_count.txt", "r") as f:
                self.N = int(f.read())
            with open("doc_id_map.json", "r") as f:
                self.doc_id_map = json.load(f)
            self.id_to_filename = {str(v): k for k, v in self.doc_id_map.items() if k.endswith('.txt')}
        except FileNotFoundError as e:
            print(f"Error: Could not find necessary index file. {e}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON in index file. {e}")
            exit(1)

    def search(self, query, top_k=10):
        query_terms = [self.stemmer.stem(term.lower()) for term in word_tokenize(query)]
        query_weights = self.compute_query_weights(query_terms)
        
        scores = defaultdict(float)
        for term, weight in query_weights.items():
            if term in self.dictionary:
                for doc_id, log_tf in self.dictionary[term]["postings"]:
                    if str(doc_id) in self.id_to_filename:  # Only consider .txt files
                        scores[str(doc_id)] += weight * log_tf  # lnc for documents (no idf)

        # Normalize scores by document length
        for doc_id in scores:
            scores[doc_id] /= float(self.doc_lengths.get(doc_id, 1))

        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], int(x[0])))
        results = []
        for doc_id, score in sorted_scores[:top_k]:
            filename = self.id_to_filename.get(doc_id, f"Unknown_Document_{doc_id}")
            results.append((filename, score))
        return results

    def compute_query_weights(self, query_terms):
        term_freq = defaultdict(int)
        for term in query_terms:
            term_freq[term] += 1

        weights = {}
        for term, tf in term_freq.items():
            if term in self.dictionary:
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                idf = math.log10(self.N / self.dictionary[term]["df"])
                weights[term] = tf_weight * idf  # ltc for queries

        # Normalize weights (cosine normalization)
        norm = math.sqrt(sum(w**2 for w in weights.values()))
        if norm > 0:
            for term in weights:
                weights[term] /= norm

        return weights

if __name__ == "__main__":
    searcher = VSMSearcher()
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() in ['quit', 'exit']:
            break
        
        results = searcher.search(query)
        print(f"\nTop {len(results)} results:")
        for filename, score in results:
            print(f"{filename}, {score:.4f}")
        print()