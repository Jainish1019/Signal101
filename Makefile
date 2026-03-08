.PHONY: install download process train evaluate dashboard all

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m nltk.downloader vader_lexicon punkt_tab stopwords

download:
	python scripts/01_download_filings.py
	python scripts/02_parse_chunk.py

process:
	python scripts/03_extract_features.py
	python scripts/04_build_index.py

train:
	python scripts/05_train_models.py
	python scripts/06_score_decide.py

evaluate:
	python scripts/07_evaluate.py
	python scripts/08_run_agents.py

dashboard:
	streamlit run dashboard/app.py --server.headless true

pipeline: download process train evaluate

all: install pipeline dashboard
