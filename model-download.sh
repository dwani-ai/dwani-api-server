#!/bin/bash

# Recommended for L4 24GB: Q6_K LLM + F16 mmproj (best quality/speed balance)
mkdir -p models
cd models

export HF_HUB_ENABLE_HF_TRANSFER=1

# Official Qwen repo (reliable, good quality quants)
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q8_0.gguf --local-dir .
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf --local-dir .

echo "Download complete! Recommended command flags:"
echo "  --model /models/Qwen3VL-8B-Instruct-Q8_0.gguf"
echo "  --mmproj /models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf.gguf"

# Alternative: Even higher quality Q8_0 LLM
# huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-GGUF Qwen3VL-8B-Instruct-Q8_0.gguf --local-dir .

# Alternative: Unsloth Dynamic quants (often slightly better at same bits)
# huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF Qwen3-VL-8B-Instruct-UD-Q6_K.gguf --local-dir .
# huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF mmproj-F16.gguf --local-dir .