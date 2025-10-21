dwani.ai - API Management Server

--
docker compose -f gcp-compose.yml up -d

--


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


--

deepseek-ocr  : https://huggingface.co/deepseek-ai/DeepSeek-OCR


https://github.com/deepseek-ai/DeepSeek-OCR/



<!-- 
docker run  --env-file .env -p 80:80 dwani/api-server:latest
DOCKER_BUILDKIT=1 docker build -t slabstech/dwani-api-server .


uvicorn src.app.main:app --host 0.0.0.0 --port 8000


docker build -t dwani/api-server:latest -f Dockerfile .


docker run  --env-file .env dwani/api-server:latest

pip install vllm 

vllm serve RedHatAI/gemma-3-4b-it-FP8-dynamic --served-model-name gemma3 --host 0.0.0.0 --port 9000 --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --max-model-len 8192 --disable-log-requests --dtype bfloat16 --enable-chunked-prefill --enable-prefix-caching --max-num-batched-tokens 8192 --chat-template-content-format openai

-->


sudo apt update
sudo apt install libgirepository1.0-dev libcairo2-dev python3-dev pkg-config build-essential
