#!/usr/bin/env bash
set -e
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose build
docker-compose up
