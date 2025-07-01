dwani.ai - API Management Server


DOCKER_BUILDKIT=1 docker build -t slabstech/dwani-api-server .


uvicorn src.app.main:app --host 0.0.0.0 --port 8000


docker build -t dwani/api-server:latest -f Dockerfile .


docker run  --env-file .env dwani/api-server:latest