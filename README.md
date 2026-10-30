# MiniMax H3 Universal (picture to video with sound) — RunPod Serverless worker

A ready-to-build [RunPod Serverless](https://runpod.io?ref=qpguwapm) worker for **any [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) model on Civitai** — the fine-tunes and merges that make video with its own sound (voice, effects, music) from your picture. You paste the model's Civitai link into the `Dockerfile`; the build downloads that model and bakes it into the image, so a cold start does not re-download anything. Based on the official [`runpod/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui).

> Some Civitai models are 18+. You download them with your own Civitai account and are responsible for following each model's license ([MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)).

## What it takes

- A **Checkpoint** whose base model on Civitai is **MiniMax H3**. Other models, LoRAs and others are refused at build time with a clear message — they need a different workflow.
- One file of the whole model that plain ComfyUI loads on a 48 GB GPU: **INT8** first, else **FP8** (GGUF, NVFP4, INT4 and BF16 files are skipped). It is checked to be an H3 model inside before the download, checked by SHA-256 against Civitai, and saved as `diffusion_models/model.safetensors`; `model.json` next to it records which model and version it is.

H3's text encoder and VAEs from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) are downloaded too, pinned by SHA-256, and ComfyUI is upgraded to 0.43.1 (MiniMax H3 is built into ComfyUI from there on). LoRAs: see [LoRAs](#loras).

## Setup (about 10 minutes + build time)

New to this? Follow the [step-by-step setup guide](https://cobaltpear.github.io/guide.html).

1. **Make your own private copy** — click **Use this template** → **Create a new repository** → choose **Private**.
   (Your Civitai token will be in it, so it must stay private.)
2. **Fill in two lines** of `Dockerfile` in your copy (✏️ edit → Commit):
   - `PASTE_YOUR_CIVITAI_TOKEN_HERE` → your token (civitai.com → Account settings → **Security & Apps** → **API Keys** → **Add API key** → Permission preset **Read Only**).
   - `PASTE_THE_MODEL_LINK_HERE` → the model's page link, e.g. `https://civitai.com/models/2877206`. A link without `modelVersionId` takes the newest version; a plain version number works too.

   No RunPod account yet? [Sign up here](https://runpod.io?ref=qpguwapm) — a referral link: after you first add $10, RunPod gives you a bonus credit (usually $5–10) and this repo a small credit.
3. **Check the model's settings** — see [Settings](#settings) below. Most models need nothing here.
4. **Create the endpoint** — see [Create the endpoint](#create-the-endpoint) below.
5. **Create an API key** — RunPod → Settings → API Keys → **Restricted**, allow only this endpoint.

One endpoint = one model. For another model, make another copy (or change the link and rebuild).

## Settings

Each model's Civitai page says which settings its creator recommends. This repo has two:

| The creator recommends | Use | In your `Dockerfile` |
|---|---|---|
| no sparse attention (most models — e.g. 8 steps, Euler) | `pack.json` — 8 steps, Euler, shift 8/4 | leave `ARG RECOMMENDED=0` |
| **sparse attention** (e.g. «90% sparse from the start») | `pack-sparse.json` — 90% sparse from the first step, shift 12/3, res_multistep, 8 steps | change to `ARG RECOMMENDED=1` |

`RECOMMENDED=1` installs PyTorch for CUDA 13, which the fast sparse attention needs. It only runs on machines with CUDA 13, so also set **Minimum CUDA version** `13.0` on the endpoint (below). It downloads more, so the build runs close to RunPod's 30-minute limit. A sparse model also runs with `RECOMMENDED=0` — just slower.

Other numbers on the model's page (steps, shift, sampler) go in the workflow — change them in your client, or in the JSON you send.

## Create the endpoint

RunPod → **Serverless** → **New Endpoint** → **Deploy from a GitHub repository** → pick your copy (the first time, **Edit connection** lets RunPod see it). Then, top to bottom on the page:

| Where on the page | What to set |
|---|---|
| **Endpoint name** | anything — your copy's name is fine |
| **GPU configuration** | tick every box of 48 GB or more, on both tabs **Previous generation** and **Latest generation** (more boxes = easier to get a GPU) |
| **Max workers** | `1` |
| Click **Advanced settings** to open it: | |
| → **Active workers** | `0` — nothing to pay while idle |
| → **GPU count** | `1` |
| → **Idle timeout** | `5` sec — or longer if you make several in a row (see below) |
| → **Enable FlashBoot** | on (the switch) · **Standard** |
| → **Enable execution timeout** | on · **Execution timeout** `900` sec |
| → **Minimum CUDA version** | leave it as it is — only with `RECOMMENDED=1`: `13.0` |
| → click **Container configuration** to open it → **Container disk** | 80 GB |

Leave everything else as it is, then **Deploy** at the bottom. The first build takes a while (it downloads about 40 GB).

**This build can run close to RunPod's 30-minute build limit.** How long it takes depends on RunPod's network that day. If the build stops with a time-out, nothing is wrong with your copy — build it again: in your copy, edit `README.md` (✏️), add any line, and commit. RunPod starts a new build by itself. Still timing out with `RECOMMENDED=1`? Change it back to `0` and set **Minimum CUDA version** back to what it was (`12.0`).
<!-- cp:initializing first=request -->
The very first request after that waits while a worker pulls the image — this can take a long time. The worker shows **Initializing**. To see what it's doing: endpoint page → **Workers** → click the worker → **Logs** → **System**.

- No error → leave it. A big image can sit on one line for a long time. Lines starting with `INFO` are normal.
- The same error keeps repeating → delete that worker with the 🗑 icon at the right end of its row. RunPod starts a new one.
- Still failing after that → ask us in [Discussions](https://github.com/cobaltpear/cobaltpear.github.io/discussions) with a screenshot of the log, or the whole log.
<!-- /cp:initializing -->

**Making several in a row?**

<!-- cp:several result=result -->
RunPod turns the worker off 5 seconds after each result (**Idle timeout** `5` sec). The next request then waits while a worker starts and loads the model again. It may be the same machine or a different one, so this wait can be short or long.

If you look at a result and send the next one soon after, set **Idle timeout** higher, for example `300` sec (endpoint page → **Manage** → **Edit Endpoint** → **Advanced settings**). The worker then stays on for that long after each result, and the next one starts right away. You pay for that time too, so set it back to `5` when you're done.

In short: `5` sec means you wait at each start and pay nothing while idle. Longer means no wait, and you pay while it waits for you.
<!-- /cp:several -->

## Or run it on Modal

The same worker also runs on [Modal](https://modal.com) — another serverless GPU service, billed by the second — from `modal_app.py` in this repo. It builds the same `Dockerfile` and answers the same requests as on RunPod (`/run`, `/status/<id>`), so a client that calls RunPod endpoints needs only the other address and key.

On your computer, with Python 3.10 or newer, in a copy of your repo (`git clone` it, with your `Dockerfile` lines filled in):

```sh
pip install modal
modal setup                                                   # opens a page to sign in to Modal
modal secret create cobaltpear-key CP_KEY=<a long password you make up>
modal deploy modal_app.py                                     # builds, then prints the worker's address
```

- The address ending in `-api….modal.run` is your endpoint; send `Authorization: Bearer <that password>` with each request.
- GPU: the `GPU` line at the top of `modal_app.py` (48 GB by default). A worker turns off 60 s after its last picture.
- Modal needs a payment method on your account to use GPUs. Its prices: [modal.com/pricing](https://modal.com/pricing).
- Updates: run `modal deploy modal_app.py` again. If the worker doesn't start there with `RECOMMENDED=1`, use `RECOMMENDED=0`.

## Updates

Your copy only holds the token and the link. The build steps (`build.sh`, `handler.py`) are downloaded from this repo on every build, so fixes here reach you when you rebuild.

To rebuild: commit any change to the `main` branch of your copy — RunPod builds again on every push to `main` (the endpoint's **Builds** tab shows it). Nothing to change? Edit `README.md` (✏️), add a line, **Commit changes…** → **Commit changes**.

Each change of the build files is kept as a release (`files-…`). To stay on one, replace `latest/download` with `download/files-…` (that release's name) in both `ADD` lines of your `Dockerfile`.

## Use it

Any client that sends ComfyUI API-format workflows to RunPod works. `pack.json` (or `pack-sparse.json`, see [Settings](#settings)) holds the workflow and tells a client where the prompt, seed and size go (format `pack/1`, engine `comfyui`).

Raw request:

```bash
IMG=$(base64 -w0 photo.png)
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <API_KEY>" -H "Content-Type: application/json" \
  -d "{\"input\": {\"workflow\": $(jq .workflow pack.json | jq '.["6"].inputs.string_b = "She turns her head and smiles. Soft room tone, a quiet laugh." | .["15"].inputs.noise_seed = 1'), \"images\": [{\"name\": \"start_image.png\", \"image\": \"$IMG\"}]}}"
```

Your picture goes in `images` under the name `start_image.png` — the workflow's LoadImage node reads it from there.

The video (MP4 with sound) comes back as base64 in `output.images[0].data`.

Tips: Starts from your picture: 5 s at 24 fps with sound. Describe the motion and the sound, not the picture. The workflow puts one line in front of your prompt that tells H3 to start from your picture.

### LoRAs

- **Per picture:** send up to 4 LoRAs with a request: `input.remote_loras` = `[{"name": "remote_<id>.safetensors", "url": "<download link>", "sha256": "<file sha256>"}]` plus a `LoraLoader` reading that name in the workflow — the worker downloads each once and keeps it while it runs. [Cobalt Pear](https://cobaltpear.github.io/) does all of this from a Civitai or Hugging Face link. For more than 4, set the environment variable `REMOTE_LORA_MAX` (e.g. `8`) on your endpoint (**Manage → Edit Endpoint → Environment Variables**).
- **Built in:** put links in the `LORAS` line of your `Dockerfile` — each followed by its strength — and rebuild. They are downloaded once at build time, so a fresh worker doesn't download them again. A request picks which ones it uses with `input.baked_loras` = `[{"link": "<link from LORAS>", "strength": 0.8}]` (`[]` = none); without it every one is used at its `LORAS` strength. One that isn't built in (yet) is left out and named in `output.warning`, and `output.built_in` lists the built-in LoRAs. Put their trigger words in your prompt yourself. To accept only built-in LoRAs, set `REMOTE_LORA_MAX` to `0`.
<!-- cp:video-loras -->
- **Video model:** the worker is big, so its build is already long. Built-in LoRAs add to it, and RunPod stops a build after 30 minutes — so for this model, send LoRAs with each request (**By link** in Cobalt Pear).
<!-- /cp:video-loras -->
- Pick LoRAs made for MiniMax H3 — the worker warns (`output.warning`) when one doesn't fit.

### Compatible clients

- [Cobalt Pear](https://cobaltpear.github.io/) — a one-page web app: add this repo's URL (or, for sparse attention, `https://raw.githubusercontent.com/cobaltpear/civitai-minimax-h3/main/pack-sparse.json`), your endpoint ID and API key, then attach a picture and type the motion. Everything stays in your browser. [Try the demo](https://cobaltpear.github.io/demo/) without RunPod.

<!-- cp:page-changes -->
RunPod and Civitai change their pages from time to time. If a button or step here looks different from what you see, tell us in [Discussions](https://github.com/cobaltpear/cobaltpear.github.io/discussions) and we'll update this guide.
<!-- /cp:page-changes -->

Questions or ideas: [Discussions](https://github.com/cobaltpear/cobaltpear.github.io/discussions).

## License

AGPL-3.0 ([LICENSE](LICENSE)). This worker is based on [runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui) (AGPL-3.0), and `handler.py` is a changed copy of its handler. The models you build into it keep their own licenses.
