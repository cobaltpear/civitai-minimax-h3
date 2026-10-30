#!/bin/sh
# MiniMax H3 Universal (picture to video with sound) — everything the worker image needs. The Dockerfile in your copy downloads this file
# from the original repo on every build, so fixes made there reach you with a rebuild.
set -eu
# civitai.env in your copy wins over the two Dockerfile lines
if [ -f /setup/civitai.env ]; then
  for k in CIVITAI_TOKEN CIVITAI_MODEL; do
    v=$(sed -n "s/^[[:space:]]*$k[[:space:]]*=//p" /setup/civitai.env | tr -d '\r"'"'"' ' | head -n 1)
    [ -z "$v" ] || eval "$k=\$v"
  done
fi
export CIVITAI_TOKEN CIVITAI_MODEL
case "${CIVITAI_TOKEN:-}" in ""|PASTE_*) echo "ERROR: put your Civitai token at the top of the Dockerfile" >&2; exit 1;; esac
case "${CIVITAI_MODEL:-}" in ""|PASTE_*) echo "ERROR: put the model's Civitai page link (or version number) at the top of the Dockerfile" >&2; exit 1;; esac

# Find the one file the link means and check it is a MiniMax H3 model before downloading it (its tensor names are read
# from the file's first bytes). It is always saved as diffusion_models/model.safetensors — the pack's workflow loads that name.
python3 - <<'PY'
import json, os, re, struct, sys, urllib.request
token, link = os.environ["CIVITAI_TOKEN"], os.environ["CIVITAI_MODEL"].strip()
def fail(msg): sys.exit("ERROR: " + msg)
def get(path):
    req = urllib.request.Request("https://civitai.com/api/v1/" + path, headers={"Authorization": "Bearer " + token, "User-Agent": "cobaltpear"})
    try: return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e: fail(f"could not ask Civitai about {link!r} ({e})")
# the whole H3 model as one .safetensors that plain ComfyUI loads on a 48 GB GPU: INT8 first, then FP8 (not GGUF, NVFP4, INT4, BF16)
def rank(f):
    fp, name = (f.get("metadata", {}).get("fp") or "").lower(), f["name"].lower()
    if f.get("type") not in ("Model", "Diffusion Model") or f.get("metadata", {}).get("format") != "SafeTensor" or f.get("sizeKB", 0) < 15e6: return None
    if re.search(r"nvfp4|int4|gguf|mxfp", name): return None
    return 0 if fp == "int8" or "int8" in name else 1 if fp == "fp8" or "fp8" in name else None
def usable(v): return sorted((f for f in v.get("files", []) if rank(f) is not None), key=rank)
m = re.search(r"modelVersionId=(\d+)", link) or re.fullmatch(r"(\d+)", link)
if m: version = get(f"model-versions/{m.group(1)}")
else:
    m = re.search(r"/models/(\d+)", link) or fail(f"CIVITAI_MODEL {link!r} is not a Civitai model link")
    vs = get(f"models/{m.group(1)}")["modelVersions"]
    version = next((v for v in vs if usable(v)), vs[0])   # the newest version with a file this worker can load
    version = get(f"model-versions/{version['id']}")
base, kind = version.get("baseModel", ""), version.get("model", {}).get("type", "")
print(f"Civitai: {version.get('model', {}).get('name')} — {version.get('name')} (version {version['id']}, {base}, {kind})")
if kind != "Checkpoint": fail(f"this is a {kind}, not a Checkpoint")
if base != "MiniMax H3": fail(f"this is a {base} model — this worker takes MiniMax H3 models only")
files = usable(version)
if not files: fail("no INT8 or FP8 .safetensors file of the whole model in this version (GGUF, NVFP4, INT4 and BF16 files are not taken) — pick another version")
f = files[0]
url = f["downloadUrl"] + ("&" if "?" in f["downloadUrl"] else "?") + "token=" + token
def part(a, b): return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "cobaltpear"}), timeout=120).read()
n = struct.unpack("<Q", part(0, 7))[0]
keys = list(json.loads(part(8, 7 + n)))
if not (any(k.startswith("video_patch_proj") for k in keys) and any(k.startswith("audio_patch_proj") for k in keys)):
    fail(f"{f['name']} is not a MiniMax H3 model inside, though Civitai lists it as {base} — pick another version")
print(f"Downloading {f['name']} ({f.get('sizeKB', 0) / 1e6:.1f} GB)")
open("/tmp/model.url", "w").write(url)
open("/tmp/model.sha", "w").write(f.get("hashes", {}).get("SHA256", "").lower())
json.dump({"version": version["id"], "name": version.get("model", {}).get("name"), "file": f["name"], "base": base}, open("/comfyui/models/diffusion_models/model.json", "w"))
PY

# file <dest> <url> <sha256> — starts the download in the background, many connections per file, so the four big files
# (about 40 GB) arrive within RunPod's 30-minute build limit. `wait_files` waits for all of them and checks each one.
apt-get update -qq && apt-get install -y -qq aria2 >/dev/null
CHECKS="" PIDS=""
file() {
  aria2c -q -x 16 -s 16 -k 64M --file-allocation=none --max-tries=5 --retry-wait=10 -d "$(dirname "$1")" -o "$(basename "$1")" "$2" &
  PIDS="$PIDS $!"
  [ -z "$3" ] || CHECKS="$CHECKS$3  $1
"
}
wait_files() {
  for pid in $PIDS; do wait "$pid" || { echo "ERROR: a model download failed" >&2; exit 1; }; done
  printf "%s" "$CHECKS" | sha256sum -c - || { echo "ERROR: a model file doesn't match its SHA-256 — try the build again" >&2; exit 1; }
}

# your model (its SHA-256 from Civitai; a file Civitai lists without one is not checked)
file /comfyui/models/diffusion_models/model.safetensors "$(cat /tmp/model.url)" "$(cat /tmp/model.sha)"
[ -s /tmp/model.sha ] || echo "No SHA-256 listed for this file — not checked"
rm -f /tmp/model.url /tmp/model.sha

# the text encoder and VAEs every MiniMax H3 model uses — Comfy-Org/MiniMax-H3 (MiniMax H3 Community License), pinned by SHA-256 (15.7 GB)
file /comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/77aac5182eb85bb20930536bac73e498bbe2bfd7/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6

# Comfy-Org/MiniMax-H3 (2.8 GB)
file /comfyui/models/vae/minimax_h3_video_vae_int8_convrot.safetensors \
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/77aac5182eb85bb20930536bac73e498bbe2bfd7/vae/minimax_h3_video_vae_int8_convrot.safetensors" \
  52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e

# Comfy-Org/MiniMax-H3 (0.6 GB)
file /comfyui/models/vae/minimax_h3_audio_vae_fp32.safetensors \
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/77aac5182eb85bb20930536bac73e498bbe2bfd7/vae/minimax_h3_audio_vae_fp32.safetensors" \
  8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48

# ComfyUI 0.34.0 (base image) → 0.43.1 — for the MiniMax H3 nodes. Install into /opt/venv (the one ComfyUI actually runs with).
cd /comfyui
git fetch --depth 1 https://github.com/Comfy-Org/ComfyUI.git tag v0.43.1
git checkout -f v0.43.1
uv pip install --python /opt/venv/bin/python -r requirements.txt "transformers>=4.50.3,<5" "huggingface-hub<1.0"
# RECOMMENDED=1 (Dockerfile): PyTorch for CUDA 13 (cu130) — comfy-kitchen's fast kernels (sparse attention, int8) need it. Same torch version, other build.
# RECOMMENDED=0 keeps the CUDA 12.8 PyTorch: sparse attention falls back to plain attention, and any machine can run it.
if [ "${RECOMMENDED:-0}" = 1 ]; then
  TV=$(/opt/venv/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
  uv pip install --python /opt/venv/bin/python --reinstall "torch==$TV" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
fi
/opt/venv/bin/python -c "import torch, comfy_kitchen.backends.cuda as k; print('RECOMMENDED', '${RECOMMENDED:-0}', '| torch', torch.__version__, '| kitchen CUDA ext:', k._EXT_AVAILABLE, k._EXT_ERROR)"
timeout 300 /opt/venv/bin/python main.py --quick-test-for-ci --cpu

wait_files
