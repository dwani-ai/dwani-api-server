# Extend the official vLLM image
FROM vllm/vllm-openai:v0.8.5

# Set working directory (matches your original setup)
WORKDIR /app

# Install the missing dependencies
RUN pip install --no-cache-dir addict matplotlib

# No other changes needed—vLLM entrypoint remains intact