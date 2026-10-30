# MiniMax H3 Universal (picture to video with sound) — RunPod Serverless ComfyUI worker
# Build on RunPod from your own PRIVATE copy of this repo (Dockerfile path = Dockerfile, build context = .).
# The build steps are downloaded from the original repo, so you never need to edit anything but these two lines (and the optional LORAS and RECOMMENDED lines).

# >>> Put your Civitai API token here (civitai.com → Account settings → Security & Apps → API Keys → Add API key → Read Only). Keep your repo PRIVATE. <<<
ARG CIVITAI_TOKEN=PASTE_YOUR_CIVITAI_TOKEN_HERE
# >>> Put the model's Civitai page link here (or just its version number, e.g. 3392061). <<<
ARG CIVITAI_MODEL=PASTE_THE_MODEL_LINK_HERE

# >>> Optional — LoRAs built into the worker (each video picks which to use): Civitai or Hugging Face links, each followed by
#     its strength (1 when left out), separated by spaces. Leave it empty for none. Example:
#     ARG LORAS="https://civitai.com/models/12345 0.8 https://huggingface.co/user/some-lora 0.6"   <<<
ARG LORAS=""

# >>> RECOMMENDED=0 (default): plain attention — runs on any machine.
#     RECOMMENDED=1: for models whose creator recommends sparse attention — PyTorch for CUDA 13, so set the endpoint's
#     Minimum CUDA version to 13.0, and use pack-sparse.json. It builds slower, close to RunPod's 30-minute limit.   <<<
ARG RECOMMENDED=0

FROM runpod/worker-comfyui:5.10.0-base
ARG LORAS
ARG CIVITAI_TOKEN
ARG CIVITAI_MODEL
ARG RECOMMENDED
COPY . /setup/
ADD https://github.com/cobaltpear/civitai-minimax-h3/releases/latest/download/build.sh /build.sh
RUN CIVITAI_TOKEN="$CIVITAI_TOKEN" CIVITAI_MODEL="$CIVITAI_MODEL" RECOMMENDED="$RECOMMENDED" sh /build.sh && rm -rf /setup
ADD https://github.com/cobaltpear/civitai-minimax-h3/releases/latest/download/handler.py /handler.py
RUN python /handler.py --bake-loras
