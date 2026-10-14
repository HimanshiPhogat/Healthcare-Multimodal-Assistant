.PHONY: install train-text train-image test api app docker

install:
	pip install -r requirements.txt

train-text:
	python -m mmhealth.train_text

train-image:
	python -m mmhealth.train_image

test:
	pytest -q

api:
	uvicorn mmhealth.api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t mmhealth .
