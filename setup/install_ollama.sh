#!/bin/bash

curl -fsSL https://ollama.com/install.sh | sh

# Optional: Move ollama to /data
# sudo mv /usr/local/lib/ollama /data/setup/ollama
# sudo ln -s /data/setup/ollama /usr/local/lib/ollama

# systemctl edit ollama.service
# Environment="OLLAMA_MODELS=/data/setup/ollama_models"
# sudo systemctl daemon-reload
# sudo systemctl restart ollama
# sudo chown -R ollama:ollama ollama_models/

sudo systemctl stop ollama.service
sudo systemctl status ollama.service

mv "$HOME/.ollama/models" /data/setup/ollama_models
ln -s /data/setup/ollama_models "$HOME/.ollama/models"
sudo chmod -R o+rx /data/setup/ollama_models
sudo usermod -aG ollama $USER
ls -l ~/.ollama | grep models # Check symlink

sudo systemctl start ollama.service
ollama serve &