from flask import Flask, request, jsonify


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

import os
import re
import math
import pickle
import numpy as np
from collections import defaultdict, Counter
from inverted_index_gcp import InvertedIndex

GCS_BUCKET = "map_reduce_323866285"
GCS_PREFIX = "postings_gcp"

import gcsfs
fs = gcsfs.GCSFileSystem()

# a fixed set of common English stopwords that are removed from queries in order to reduce noise.
stopwords_frozen = frozenset([
    'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'as', 'by', 'at',
    'from', 'that', 'this', 'it', 'be', 'are', 'was', 'were', 'or', 'an', 'a'])

# regular expression used for tokenization, matches alphanumeric tokens
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text, stopwords_frozen):
    """
   tokenize input text by lowercasing, applying a regex-based tokenizer and removing stopwords
   """
    return [token.group() for token in RE_WORD.finditer(text.lower())
            if token.group() not in stopwords_frozen]


# GCS path in order to get files from the bucket
def gcs_path(filename: str) -> str:
    return f"{GCS_BUCKET}/{GCS_PREFIX}/{filename}"


def load_pkl_from_gcs(filename: str):
    """
    load a pickled object directly from Google Cloud Storage using gcsfs
    """
    with fs.open(gcs_path(filename), "rb") as f:
        return pickle.load(f)

# load the inverted body index object from GCS
IDX = InvertedIndex.read_index("postings_gcp", "index", bucket_name="map_reduce_323866285")
# load data structures required for ranking
DOC_LEN_DICT = load_pkl_from_gcs("doc_lengths.pkl")  # document lengths
AVGDL = load_pkl_from_gcs("avg_doc_len.pkl")   # average document length
DOCID_TO_TITLE = load_pkl_from_gcs("id_to_title.pkl")  # document ids to titles

# normalize title keys to int
if len(DOCID_TO_TITLE) > 0:
    k = next(iter(DOCID_TO_TITLE.keys()))
    if isinstance(k, str):
        DOCID_TO_TITLE = {int(doc_id): title for doc_id, title in DOCID_TO_TITLE.items()}


def normalize_posting_locs(index_obj: InvertedIndex):
    """
   normalize posting list file paths so that each path includes the correct GCS prefix exactly once
   """
    if not hasattr(index_obj, "posting_locs") or index_obj.posting_locs is None:
        return

    prefix = GCS_PREFIX.rstrip("/")
    bucket = GCS_BUCKET

    def fix_path(file_name: str) -> str:
        file_name = str(file_name).lstrip("/")
        # remove bucket name if it was accidentally included in the path
        if file_name.startswith(bucket + "/"):
            file_name = file_name[len(bucket) + 1:]
        # prevent duplicated prefixes
        double = f"{prefix}/{prefix}/"
        if file_name.startswith(double):
            file_name = file_name[len(prefix) + 1:]
        # ensure the prefix appears exactly once
        if not file_name.startswith(prefix + "/"):
            file_name = f"{prefix}/{file_name}"
        return file_name

    new_posting_locs = {}
    for term, locs in index_obj.posting_locs.items():
        new_posting_locs[term] = [(fix_path(file_name), offset) for (file_name, offset) in locs]
    index_obj.posting_locs = new_posting_locs


# apply path normalization to the loaded index
normalize_posting_locs(IDX)
# BASE_DIR is intentionally set to an empty string
# when using GCS, read_a_posting_list receives the bucket name separately and posting file paths are resolved relative to the bucket root
BASE_DIR = ""


class BM25:
    """
    initialize a BM25 ranking model.
    index: Inverted index containing term statistics and posting lists
    doc_len_dict: Dictionary mapping doc_id to document length
    avgdl: Average document length in the corpus
    k1, b: Standard BM25 hyperparameters controlling term frequency saturation and document length normalization
    """
    def __init__(self, index, doc_len_dict, avgdl, k1=1.5, b=0.75):
        self.index = index
        self.doc_len_dict = doc_len_dict
        self.avgdl = float(avgdl)
        self.k1 = k1
        self.b = b
        self.N = len(doc_len_dict)  # total number of documents

    def idf(self, term):
        """
        compute inverse document frequency (IDF) for a given term using the standard BM25 formulation.
        """
        number_doc_term_id = self.index.df.get(term, 0)
        if number_doc_term_id == 0:
            return 0.0
        return math.log(1.0 + (self.N - number_doc_term_id + 0.5) / (number_doc_term_id + 0.5))

    def search(self, query_tokens, topN=100):
        """
       rank documents using the BM25 scoring function
       query_tokens: Tokenized query after stopword removal
       topN: Number of top-ranked documents to return
       """
        scores = defaultdict(float)
        # check empty index or invalid average document length
        if self.N == 0 or self.avgdl == 0:
            return []
        # term frequency in the query
        term_frequency_query = Counter(query_tokens)
        for term, query_frequency in term_frequency_query.items():
            # skip terms that do not appear in the index
            if term not in self.index.df:
                continue
            term_idf = self.idf(term)
            # read posting list for the term from GCS
            posting_list = self.index.read_a_posting_list(
                BASE_DIR, term, bucket_name=GCS_BUCKET)
            for doc_id, tf in posting_list:
                doc_length = self.doc_len_dict.get(doc_id, 0)
                if doc_length == 0:
                    continue
                # BM25 scoring formula
                numerator = term_idf * tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / self.avgdl))
                # multiply by query term frequency (qf) to account for repeated terms
                scores[doc_id] += (numerator / denominator) * query_frequency
        # return top N documents sorted by score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topN]


BM25_ENGINE = BM25(IDX, DOC_LEN_DICT, AVGDL, k1=1.5, b=0.75)


@app.route("/search")
def search():
    """
    main search endpoint.
    receives a query string, applies tokenization and BM25 ranking and returns a ranked list of (doc_id, title) pairs.
    """
    search_results = []
    query = request.args.get('query', '')
    # check query length
    if len(query) == 0:
        return jsonify(search_results)
    # tokenize query and remove stopwords
    tokens = tokenize(query, stopwords_frozen)
    # check number of tokens
    if len(tokens) == 0:
        return jsonify([])
    # rank documents using BM25
    ranked = BM25_ENGINE.search(tokens, topN=100)
    # return document IDs and titles
    search_results = [(int(doc_id), DOCID_TO_TITLE.get(int(doc_id), str(doc_id)))
           for doc_id, _ in ranked]

    return jsonify(search_results)


@app.route("/search_body")
def search_body():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
