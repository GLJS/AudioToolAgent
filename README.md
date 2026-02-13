# AudioToolAgent

Code release for the **AudioToolAgent** paper. See paper here: https://arxiv.org/abs/2510.02995

The repository exposes a language-agent scaffold that calls audio specialists as tools. Two ready-to-run configurations are provided:

- **AudioToolAgent-Closed**: Gemini 3 Pro orchestrator with Google Gemini 3 Pro tool, Qwen2.5 Omni, AudioFlamingo, and Whisper.
- **AudioToolAgent-Open**: Qwen3-235B orchestrator (via Chutes AI with OpenRouter fallback) with Qwen3 Instruct, Qwen2.5 Omni, AudioFlamingo, and Whisper.


## Repository Layout

- [`audiotoolagent/`](audiotoolagent/) — core package: agent runtime, tools, APIs
  - [`agent.py`](audiotoolagent/agent.py) — main agent implementation
  - [`config.py`](audiotoolagent/config.py) — YAML config loader
  - [`tools/`](audiotoolagent/tools/) — tool adapters (local + API)
  - [`apis/`](audiotoolagent/apis/) — FastAPI servers for local tools
- [`configs/`](configs/) — example configs
  - [`audiotoolagent.yaml`](configs/audiotoolagent.yaml)
  - [`audiotoolagent_open.yaml`](configs/audiotoolagent_open.yaml)
- [`Evaluation/`](Evaluation/) — benchmark runners
  - [`MMAU_Closed.py`](Evaluation/MMAU_Closed.py) (e.g. `python -m Evaluation.MMAU_Closed --limit 50`)
  - [`MMAU_Open.py`](Evaluation/MMAU_Open.py)
  - [`MMAR_Closed.py`](Evaluation/MMAR_Closed.py)
  - [`MMAR_Open.py`](Evaluation/MMAR_Open.py)
  - [`MMAUPro_Closed.py`](Evaluation/MMAUPro_Closed.py)
  - [`MMAUPro_Open.py`](Evaluation/MMAUPro_Open.py)
- [`scripts/`](scripts/)
  - [`launch_closed.sh`](scripts/launch_closed.sh) — start local services for Gemini 3 Pro config
  - [`launch_open.sh`](scripts/launch_open.sh) — start local services for Qwen3 config
- [`main.py`](main.py) — CLI for single-run inference

## Installation

```bash
# Clone and enter the project
git clone https://github.com/GLJS/AudioToolAgent.git
cd AudioToolAgent

# Create environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Environment variables

Create a `.env` / export the following (only the relevant keys for your configuration are required):

```
# Shared
export TMP_DIR=/tmp/audiotoolagent

# Closed configuration (Gemini 3 Pro)
export GOOGLE_API_KEY="..."           # Gemini 3 Pro orchestrator + Gemini 3 Pro tool

# Open configuration (Qwen3-235B)
export CHUTES_API_KEY="..."           # Chutes AI for Qwen3-235B orchestrator
export OPENROUTER_API_KEY="..."       # OpenRouter fallback + Voxtral API

```

## Running local services

Both configurations rely on local HTTP endpoints for some tools. Two helper scripts launch the required processes and keep logs under `logs/`.

```bash
# Open / Qwen3 configuration
./scripts/launch_open.sh

# Closed / Gemini 3 Pro configuration
./scripts/launch_closed.sh
```

The scripts spawn the following components:

- **Open**: Qwen2.5 Omni (vLLM on port 4002), Qwen3 Instruct FastAPI (port 4014), and the AudioFlamingo 3 FastAPI proxy (port 4010). Whisper runs in-process via `faster-whisper`.
- **Closed**: Qwen2.5 Omni (port 4002) and the AudioFlamingo 3 FastAPI proxy (port 4010).

`hostnames.txt` is updated automatically so the tool adapters discover the correct endpoints.

## Single-run inference

Use `main.py` to run the full tool-calling pipeline for a question + audio file.

```bash
python main.py \
  --config configs/audiotoolagent.yaml \
  --audio /path/to/audio.wav \
  --question "What instrument is playing?" \
  --options "Piano" "Guitar" "Violin" "Drums"
```

Add `--no-stream` to disable incremental console streaming and `--output result.json` to save the response.

## Benchmark scripts

Each benchmark/configuration pair has its own script under `Evaluation/` so that commands from the paper can be reproduced exactly. Run them as Python modules to keep relative imports working, and use `--limit` for quick tests.

```bash
# MMAU (closed configuration)
python -m Evaluation.MMAU_Closed --limit 50

# MMAU (open configuration)
python -m Evaluation.MMAU_Open --limit 50

# MMAR (closed configuration)
python -m Evaluation.MMAR_Closed --limit 50

# MMAU-Pro (open configuration)
python -m Evaluation.MMAUPro_Open --limit 25
```

Each runner downloads the corresponding Hugging Face dataset on first use and writes optional JSON outputs when `--output` is provided.

## Configurations

Configuration files live in `configs/` and describe the orchestrator plus the set of enabled tools. Duplicate the YAMLs to experiment with alternative tool suites or decoding parameters.

Key fields:

- `orchestrator.llm_type`: `google` (Gemini 3 Pro), `chutes` (Qwen3-235B), `openrouter`, `openai`, `vllm`, etc. Use `llm_url` and `api_key_env` to point to custom endpoints.
- `tools`: ordered list of tool descriptors. Set `enabled: false` to disable a tool quickly.

## Extending the framework

- Add new tools under `audiotoolagent/tools/` by subclassing `AudioAnalysisModelTool`, `AudioTranscriptionModelTool`, or `ExternalAPITool`.
- Register new FastAPI adapters in `audiotoolagent/apis/` if a model needs to be exposed over HTTP.
- Reference the tool in a configuration file and rerun the desired evaluation script.

## Citation

If you use this codebase, please cite the AudioToolAgent paper.

```
@misc{wijngaard2025audiotoolagentagenticframeworkaudiolanguage,
      title={AudioToolAgent: An Agentic Framework for Audio-Language Models},
      author={Gijs Wijngaard and Elia Formisano and Michel Dumontier},
      year={2025},
      eprint={2510.02995},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.02995},
}
```
