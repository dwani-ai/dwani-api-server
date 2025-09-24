dwani.ai - API Management Server


sudo apt-get install poppler-utils -y

python -m venv venv
source venv/bin/activate


pip install -r requirements.txt


uvicorn src.server.main:app --host 0.0.0.0 --port 18888 

--

docker compose -f app-compose.yml up -d


docker build -t dwani/api-server:latest -f app.Dockerfile .

docker push dwani/api-server:latest


--


docker build -t dwani/api-server-nginx:latest -f Dockerfile .

 docker compose -f docker-compose.yml  up -d

<!-- 
docker run  --env-file .env -p 80:80 dwani/api-server:latest
DOCKER_BUILDKIT=1 docker build -t slabstech/dwani-api-server .


uvicorn src.app.main:app --host 0.0.0.0 --port 8000


docker build -t dwani/api-server:latest -f Dockerfile .


docker run  --env-file .env dwani/api-server:latest
-->


sudo apt update
sudo apt install libgirepository1.0-dev libcairo2-dev python3-dev pkg-config build-essential
