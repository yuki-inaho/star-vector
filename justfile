set shell := ["bash", "-cu"]

# StarVector Gradio demo helper commands
# Usage:
#   just controller                     # start controller (default :10000)
#   just worker                         # start worker for 1B model on CUDA, port 40000
#   just gradio                         # start Gradio UI on port 7000
#   just worker device=cpu              # run worker on CPU (slow)
#   just worker model_name=starvector/starvector-8b-im2svg port=41000
#   just worker-1b                      # shorthand for 1B worker (CUDA, port 40000)
#   just worker-8b                      # shorthand for 8B worker (CUDA, port 41000)
#   just controller controller_port=12000 && just worker controller_port=12000 && just gradio controller_port=12000
# Notes:
#   - Run each command in its own terminal (order: controller → worker → gradio).
#   - Ports and model names are adjustable via variables.
#   - Uses `uv run ...` to rely on your uv-managed environment.

token_cmd := "[ -f ~/.bashrc ] && source ~/.bashrc >/dev/null 2>&1; printf \"%s\" \"${HUGGINGFACE_HUB_TOKEN:-${HF_HUB_TOKEN:-${HF_TOKEN:-}}}\""

controller_host := "0.0.0.0"
controller_port := "10000"

worker_host := "0.0.0.0"
worker_port := "40000"

gradio_port := "7000"

controller:
    uv run python -m starvector.serve.controller \
      --host {{controller_host}} \
      --port {{controller_port}}

worker model_name="starvector/starvector-1b-im2svg" model_path="starvector/starvector-1b-im2svg" port=worker_port device="cuda":
    # Pull HF_TOKEN from a login shell without touching the current venv/session.
    TOKEN="$(bash -lc '{{token_cmd}}')" HF_TOKEN="$TOKEN" HF_HUB_TOKEN="$TOKEN" HUGGINGFACE_HUB_TOKEN="$TOKEN" uv run python -m starvector.serve.model_worker \
      --host {{worker_host}} \
      --controller http://localhost:{{controller_port}} \
      --port {{port}} \
      --worker http://localhost:{{port}} \
      --model-path {{model_path}} \
      --model-name {{model_name}} \
      --device {{device}}

worker-1b device="cuda" port=worker_port:
    TOKEN="$(bash -lc '{{token_cmd}}')" HF_TOKEN="$TOKEN" HF_HUB_TOKEN="$TOKEN" HUGGINGFACE_HUB_TOKEN="$TOKEN" uv run python -m starvector.serve.model_worker \
      --host {{worker_host}} \
      --controller http://localhost:{{controller_port}} \
      --port {{port}} \
      --worker http://localhost:{{port}} \
      --model-path starvector/starvector-1b-im2svg \
      --model-name starvector/starvector-1b-im2svg \
      --device {{device}}

worker-8b device="cuda" port="41000":
    TOKEN="$(bash -lc '{{token_cmd}}')" HF_TOKEN="$TOKEN" HF_HUB_TOKEN="$TOKEN" HUGGINGFACE_HUB_TOKEN="$TOKEN" uv run python -m starvector.serve.model_worker \
      --host {{worker_host}} \
      --controller http://localhost:{{controller_port}} \
      --port {{port}} \
      --worker http://localhost:{{port}} \
      --model-path starvector/starvector-8b-im2svg \
      --model-name starvector/starvector-8b-im2svg \
      --device {{device}}

gradio port=gradio_port:
    uv run python -m starvector.serve.gradio_web_server \
      --controller http://localhost:{{controller_port}} \
      --model-list-mode reload \
      --port {{port}}
