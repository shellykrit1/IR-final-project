# Wikipedia Search Engine

This repository implements a full text search engine over the English Wikipedia corpus. The system is based on an inverted index stored entirely on Google Cloud Storage and uses a BM25 based ranking approach for document retrieval. The search engine is exposed via a Flask based HTTP server and deployed on a Google Compute Engine virtual machine.

The project was developed as part of an academic Information Retrieval assignment, with emphasis on correctness, clarity, and cloud based scalability.

The repository contains the following files.

inverted_index_gcp.py
This file implements the inverted index used by the search engine. The index is designed to operate directly on data stored in Google Cloud Storage. Posting lists and all auxiliary index data are stored in binary files in GCS and accessed during query processing. The implementation supports large scale posting lists by splitting them across multiple files. All required index data is loaded from Google Cloud Storage, and the system does not rely on locally stored index files.

search_frontend.py
This file implements the search frontend using the Flask framework. The frontend loads the inverted index and all required data structures directly from Google Cloud Storage, processes incoming user queries, and ranks documents using a BM25 based scoring model. Query processing includes lowercasing, regular expression based tokenization, and stopword removal, and is consistent with the preprocessing used during index construction. The search endpoint returns a ranked list of document id and title pairs in JSON format.

startup_script_gcp.sh
This file is a startup script that is executed automatically when a Google Compute Engine virtual machine is created. The script installs all required system dependencies, creates a Python virtual environment, and installs the necessary Python packages. This ensures that the virtual machine is fully configured and ready to run the search frontend without manual intervention.

run_frontend_in_gcp.sh
This file is a deployment and execution script for Google Cloud Platform. It automates the deployment process by allocating a static external IP address, configuring firewall rules, provisioning a Compute Engine instance, uploading the search frontend code to the virtual machine, and running the Flask server in the background.

queries_train.json
This file contains training queries and relevance judgments used for offline evaluation and experimentation. Each entry consists of a natural language query and a list of Wikipedia document ids considered relevant to that query. This file is not required to run the search engine, but is used for evaluation and analysis of retrieval quality.

run_frontend_in_colab.ipynb
This Jupyter notebook is intended for running and testing the search frontend in a Google Colab environment. It allows development and debugging without deploying a Google Compute Engine virtual machine, while still loading the inverted index and all required data directly from Google Cloud Storage.

After deployment, the search engine can be queried via HTTP by providing a free text query. The system responds with a JSON formatted list of ranked document id and title pairs.

The system was implemented in Python and deployed using Google Cloud Platform services, including Google Cloud Storage and Google Compute Engine, as part of an academic Information Retrieval assignment.
