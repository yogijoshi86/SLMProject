#!/usr/bin/env bash
# One-time setup for a Colab GPU runtime (connected from VSCode or in-browser).
# Installs the model/data deps missing from the base image, then verifies GPU + login.
set -e

echo "==> Installing project + GPU/model extras"
pip install -q -e ".[quant,explainer,dev]"

echo "==> Verifying GPU"
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU — switch Colab runtime to GPU (T4)'; print('GPU:', torch.cuda.get_device_name(0))"

echo "==> Hugging Face login (needed for gated meta-llama/Llama-Guard-3-8B)"
echo "    Run:  huggingface-cli login   (paste a token with access to the Llama-Guard repo)"
echo "    And accept the license at https://huggingface.co/meta-llama/Llama-Guard-3-8B"

echo "==> Set your explainer key, e.g.:  export OPENAI_API_KEY=sk-..."
echo "Setup complete. Next: python scripts/01_extract.py --config config/colab_smoke.yaml"
