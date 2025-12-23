# Run your app normally (no sudo needed)
uvicorn src.server.main:app --host 0.0.0.0 --port 8000

# In another terminal (as root), set up redirection (persists until reboot)
sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000
sudo iptables -t nat -A OUTPUT -p tcp -o lo --dport 80 -j REDIRECT --to-port 8000  # Optional: for localhost access

export DWANI_API_BASE_URL_TTS="http://127.0.0.1:7864"
export DWANI_API_BASE_URL_ASR="http://127.0.0.1:7863"


nohup python src/multi-lingual/asr_api.py --port 7863 --host 0.0.0.0 --device cuda > asr.log 2>&1 &

nohup uvicorn src.server.main:app --host 0.0.0.0 --port 18888 > api-server.log 2>&1 &

nohup python src/gh200/main.py --host 0.0.0.0 --port 7864 --config config_two > tts.log 2>&1 &