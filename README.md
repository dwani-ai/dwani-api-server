dwani.ai - API Management Server

```bash

docker build -t dwani/api-server:latest -f Dockerfile .


docker compose -f compose.yml up -d

```



Local Run

```bash

sudo apt-get install poppler-utils -y

python -m venv venv
source venv/bin/activate


pip install -r requirements.txt


uvicorn src.server.main:app --host 0.0.0.0 --port 18888 

```