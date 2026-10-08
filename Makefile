COMPOSE = docker compose
SAMPLE ?= src/sample_data/3D Drawing Example.pdf

.PHONY: up up-airflow down logs test smoke train-smoke train build-ml push

up:            ## inference stack
	$(COMPOSE) up -d --build

up-airflow:    ## inference stack + Airflow executor
	PIPELINE_EXECUTOR=airflow $(COMPOSE) --profile airflow up -d --build

down:
	$(COMPOSE) --profile airflow --profile training down

logs:
	$(COMPOSE) logs -f api worker

test:          ## unit tests inside the service image
	$(COMPOSE) build api
	docker run --rm -v "$(PWD)/tests:/app/tests:ro" -v "$(PWD)/src/Training:/app/src/Training:ro" \
		-e PYTHONPATH=/app/src:/app/src/Deployment:/app/src/Training drawing-ai/service:latest \
		pytest -q -p no:cacheprovider /app/tests

smoke:         ## submit the sample drawing end-to-end
	./scripts/smoke_test.sh "$(SAMPLE)"

train-smoke:   ## tiny synthetic training run (CPU)
	$(COMPOSE) --profile training run --rm trainer all --raw /data/raw --dataset /data/processed/smoke --synthetic 12 --epochs 1 --batch 2

train:         ## full training on ./data/raw
	$(COMPOSE) --profile training run --rm trainer all --raw /data/raw --dataset /data/processed/latest

build-ml:      ## service image with trained-model backends
	INSTALL_ML=true $(COMPOSE) build api worker

push:          ## push images to Azure Container Registry (ACR_NAME=...)
	./scripts/push_acr.sh
