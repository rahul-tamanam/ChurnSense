.PHONY: setup ingest features train score api test clean

setup:
	pip install -r requirements.txt
	cp .env.example .env
	@echo "Edit .env and add your ANTHROPIC_API_KEY"

ingest:
	python src/ingestion/load_to_warehouse.py

features:
	python src/features/run_feature_pipeline.py
	python src/features/change_point_detection.py

train:
	python src/models/churn_model.py --train
	python src/models/uplift_model.py --train

score:
	python pipelines/scoring_pipeline.py

pipeline:
	python pipelines/full_pipeline.py

api:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

clean:
	rm -f data/warehouse/churn.duckdb
	rm -f data/exports/*.csv
	rm -rf models/
