# Wikipedia Search Engine

This repository implements a full text search engine over the Wikipedia corpus using an Inverted Index stored on Google Cloud Storage and a BM25 ranking model. The system is exposed via a Flask server and deployed on a Google Compute Engine virtual machine. The project was developed as part of an academic Information Retrieval assignment, with emphasis on correctness, clarity, and cloud-based scalability.

The repository contains the following files:

- inverted_index_gcp.py  
  Implements the core Inverted Index data structure. This file is responsible for building the index from tokenized documents, tracking document frequency and total term frequency, and writing posting lists to disk. Posting lists are stored in binary format using fixed size records. To support large-scale indexing, posting lists are split across multiple files using MultiFileWriter and MultiFileReader. Only metadata and posting locations are kept in memory and serialized using pickle, while the posting lists themselves are read from GCS on demand.

- search_frontend.py  
  Implements the search frontend using Flask. This file loads the inverted index from GCS, normalizes posting list paths, tokenizes incoming queries, and ranks documents using the BM25 scoring function. Tokenization includes lowercasing, regex-based token extraction, and stopword removal, and is consistent with the tokenization used during index construction. The main implemented endpoint is search function, which returns a ranked list of (doc_id, title) pairs.

- startup_script_gcp.sh  
  A startup script executed automatically when the Google Compute Engine virtual machine is created. The script installs system dependencies, creates a Python virtual environment, and installs all required Python packages (Flask, NumPy, Pandas, google-cloud-storage, gcsfs, Werkzeug, etc.). This ensures the VM is fully configured and ready to run the search frontend without manual intervention.

- run_frontend_in_gcp.sh  
  A deployment and execution script for Google Cloud Platform. This script allocates a static external IP address, creates a firewall rule allowing traffic on port 8080, provisions a Compute Engine instance, uploads the search_frontend.py file to the VM, and runs the Flask server using nohup.

- queries_train.json  
  Contains training queries and relevance judgments used for evaluation and experimentation. Each key is a natural language query, and each value is a list of Wikipedia document IDs considered relevant to that query. This file is not required to run the search server, but is intended for offline evaluation, ranking analysis, and metric computation such as Precision@k.

- run_frontend_in_colab.ipynb  
  This Jupyter notebook is intended for running and testing the search frontend inside Google Colab. It provides a lightweight, local-like environment for development, debugging, and validation without deploying a Google Compute Engine VM. The notebook installs all required dependencies, loads the inverted index and auxiliary data structures from Google Cloud Storage, and launches the Flask search server inside the Colab runtime.
  
After deployment, the search engine can be queried via HTTP using the following format:

http://<EXTERNAL_IP>:8080/search?query=example

The system returns a JSON response containing ranked (doc_id, title) pairs.

Developed as part of an academic Information Retrieval assignment using Python and Google Cloud Platform.
