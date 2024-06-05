# these will speed up builds, for docker-compose >= 1.25
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

all: build test

build:
	docker-compose build

logs:
	docker-compose up -d app
	docker-compose logs app | tail -100

test:
	docker-compose up -d app
	docker-compose exec app pytest -s --tb=short tests/env_test.py

black:
	black -l 86 $$(find * -name '*.py')
