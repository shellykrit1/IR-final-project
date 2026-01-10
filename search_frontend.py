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
from collections import defaultdict
from collections import Counter
from inverted_index_gcp import InvertedIndex

# ---------- GCS CONFIG ----------
# Bucket name in your case (as shown in the screenshot) is: map_reduce_323866285
# You can keep it hardcoded (recommended to avoid env mistakes)
GCS_BUCKET = "map_reduce_323866285"

# Folder INSIDE the bucket that contains:
# index.pkl, doc_lengths.pkl, avg_doc_len.pkl, id_to_title.pkl and the .bin posting files
GCS_PREFIX = "postings_gcp"
# -------------------------------

import gcsfs
fs = gcsfs.GCSFileSystem()

stopwords_frozen = frozenset([
    'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'as', 'by', 'at',
    'from', 'that', 'this', 'it', 'be', 'are', 'was', 'were', 'or', 'an', 'a'
])

# --------- tokenizer ----------
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text, stopwords_frozen):
    return [token.group() for token in RE_WORD.finditer(text.lower())
            if token.group() not in stopwords_frozen]


# --------- helpers: load pickles from GCS ----------
def gcs_path(filename: str) -> str:
    # gcsfs path format: "<bucket>/<path>"
    return f"{GCS_BUCKET}/{GCS_PREFIX}/{filename}"


def load_pkl_from_gcs(filename: str):
    with fs.open(gcs_path(filename), "rb") as f:
        return pickle.load(f)


# --------- load once (FROM GCS) ----------
# Load the index object itself from index.pkl in GCS
IDX = InvertedIndex.read_index("postings_gcp", "index", bucket_name="map_reduce_323866285")

DOC_LEN_DICT = load_pkl_from_gcs("doc_lengths.pkl")
AVGDL = load_pkl_from_gcs("avg_doc_len.pkl")
DOCID_TO_TITLE = load_pkl_from_gcs("id_to_title.pkl")

# normalize title keys to int (important)
if len(DOCID_TO_TITLE) > 0:
    k = next(iter(DOCID_TO_TITLE.keys()))
    if isinstance(k, str):
        DOCID_TO_TITLE = {int(doc_id): title for doc_id, title in DOCID_TO_TITLE.items()}

# attach DL to index for convenience (optional)
IDX.DL = DOC_LEN_DICT

# --------- IMPORTANT: make posting_locs point to the right BIN paths in GCS ----------
# In the staff code, read_a_posting_list(base_dir, term) uses base_dir as the BUCKET NAME,
# and posting_locs contain relative paths to the .bin files inside that bucket.
# We normalize them so they include the GCS_PREFIX folder ("postings_gcp").

def _needs_prefix(fname: str) -> bool:
    return not fname.startswith(GCS_PREFIX + "/")


def _normalize_posting_locs(index_obj: InvertedIndex):
    if not hasattr(index_obj, "posting_locs") or index_obj.posting_locs is None:
        return

    prefix = GCS_PREFIX.rstrip("/")
    bucket = GCS_BUCKET

    def fix(fname: str) -> str:
        fname = str(fname).lstrip("/")

        # אם שם הבאקט בטעות נכנס לתוך ה-fname, להסיר אותו
        if fname.startswith(bucket + "/"):
            fname = fname[len(bucket) + 1:]

        # למנוע כפילות postings_gcp/postings_gcp
        double = f"{prefix}/{prefix}/"
        if fname.startswith(double):
            fname = fname[len(prefix) + 1:]

        # להבטיח שיש prefix פעם אחת בדיוק
        if not fname.startswith(prefix + "/"):
            fname = f"{prefix}/{fname}"

        return fname

    new_posting_locs = {}
    for term, locs in index_obj.posting_locs.items():
        new_posting_locs[term] = [(fix(f), off) for (f, off) in locs]

    index_obj.posting_locs = new_posting_locs



_normalize_posting_locs(IDX)

# This is the key change:
# BASE_DIR for read_a_posting_list should be the BUCKET name (not ".")
BASE_DIR = ""


class BM25:
    def __init__(self, index, doc_len_dict, avgdl, k1=1.5, b=0.75):
        self.index = index
        self.doc_len_dict = doc_len_dict
        self.avgdl = float(avgdl)
        self.k1 = k1
        self.b = b
        self.N = len(doc_len_dict)

    def idf(self, term):
        n_ti = self.index.df.get(term, 0)
        if n_ti == 0:
            return 0.0
        return math.log(1.0 + (self.N - n_ti + 0.5) / (n_ti + 0.5))

    def search(self, query_tokens, topN=100):
        scores = defaultdict(float)
        if self.N == 0 or self.avgdl == 0:
            return []

        qtf = Counter(query_tokens)  # תדירות מונחים בשאילתה

        for term, qf in qtf.items():
            if term not in self.index.df:
                continue

            term_idf = self.idf(term)

            pl = self.index.read_a_posting_list(
                BASE_DIR, term, bucket_name=GCS_BUCKET
            )

            for doc_id, tf in pl:
                dl = self.doc_len_dict.get(doc_id, 0)
                if dl == 0:
                    continue

                numerator = term_idf * tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))

                # 👈 כאן ההבדל: מכפילים ב-qf
                scores[doc_id] += (numerator / denominator) * qf

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topN]


BM25_ENGINE = BM25(IDX, DOC_LEN_DICT, AVGDL, k1=1.5, b=0.75)


@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = tokenize(query, stopwords_frozen)
    if len(tokens) == 0:
        return jsonify([])

    ranked = BM25_ENGINE.search(tokens, topN=100)

    res = [(int(doc_id), DOCID_TO_TITLE.get(int(doc_id), str(doc_id)))
           for doc_id, _ in ranked]

    return jsonify(res)


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
