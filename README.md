# AudioToolAgent

Code release for the **AudioToolAgent** paper. See paper here: https://arxiv.org/abs/2510.02995

The repository exposes a language-agent scaffold that calls audio specialists as Model Context Protocol tools. Two ready-to-run configurations are provided:

- **AudioToolAgent-GPT5** (closed): GPT-5 orchestrator with vendor APIs (OpenAI GPT-4o, Google Gemini 2.5 Flash, Mistral Voxtral) plus a local AudioFlamingo server.
- **AudioToolAgent-Open** (open): DeepSeek-V3.1 orchestrator with Whisper (faster-whisper), Voxtral, Qwen2.5 Omni, DeSTA 2.5, and AudioFlamingo 3.


## Repository Layout

- [`audiotoolagent/`](audiotoolagent/) — core package: agent runtime, tools, APIs  
  - [`agent.py`](audiotoolagent/agent.py) — main agent implementation  
  - [`config.py`](audiotoolagent/config.py) — YAML config loader  
  - [`tools/`](audiotoolagent/tools/) — tool adapters (local + API)  
  - [`apis/`](audiotoolagent/apis/) — FastAPI servers for local tools  
- [`configs/`](configs/) — example configs  
  - [`closed_gpt5.yaml`](configs/closed_gpt5.yaml)  
  - [`open_deepseek.yaml`](configs/open_deepseek.yaml)  
- [`Evaluation/`](Evaluation/) — benchmark runners  
  - [`MMAU_GPT5.py`](Evaluation/MMAU_GPT5.py) (e.g. `python -m Evaluation.MMAU_GPT5 --limit 50`)  
  - [`MMAU_Open.py`](Evaluation/MMAU_Open.py)  
  - [`MMAR_GPT5.py`](Evaluation/MMAR_GPT5.py)  
  - [`MMAR_Open.py`](Evaluation/MMAR_Open.py)  
  - [`MMAUPro_GPT5.py`](Evaluation/MMAUPro_GPT5.py)  
  - [`MMAUPro_Open.py`](Evaluation/MMAUPro_Open.py)  
- [`scripts/`](scripts/)  
  - [`launch_closed.sh`](scripts/launch_closed.sh) — start local services for GPT-5 config  
  - [`launch_open.sh`](scripts/launch_open.sh) — start local services for DeepSeek config  
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

# Closed configuration (GPT-5)
export OPENAI_API_KEY="..."           # GPT-5 orchestrator + GPT-4o tool
export GOOGLE_API_KEY="..."           # Gemini 2.5 Flash
export MISTRAL_API_KEY="..."          # Voxtral API

```

## Running local services

Both configurations rely on local HTTP endpoints for some tools. Two helper scripts launch the required processes and keep logs under `logs/`.

```bash
# Open / DeepSeek configuration
./scripts/launch_open.sh

# Closed / GPT-5 configuration
./scripts/launch_closed.sh
```

The scripts spawn the following components:

- **Open**: Qwen2.5 Omni (vLLM on port 4002), DeSTA 2.5 FastAPI (port 4004), and the AudioFlamingo 3 FastAPI proxy (port 4010). Whisper runs in-process via `faster-whisper`; Voxtral is accessed via API.
- **Closed**: Qwen2.5 Omni (port 4002) and the AudioFlamingo 3 FastAPI proxy (port 4010).

`hostnames.txt` is updated automatically so the tool adapters discover the correct endpoints.

## Single-run inference

Use `main.py` to run the full tool-calling pipeline for a question + audio file.

```bash
python main.py \
  --config configs/open_deepseek.yaml \
  --audio /path/to/audio.wav \
  --question "What instrument is playing?" \
  --options "Piano" "Guitar" "Violin" "Drums"
```

Add `--no-stream` to disable incremental console streaming and `--output result.json` to save the response.

## Benchmark scripts

Each benchmark/configuration pair has its own script under `Evaluation/` so that commands from the paper can be reproduced exactly. Run them as Python modules to keep relative imports working, and use `--limit` for quick tests.

```bash
# MMAU (GPT-5 configuration)
python -m Evaluation.MMAU_GPT5 --limit 50

# MMAU (open configuration)
python -m Evaluation.MMAU_Open --limit 50

# MMAR (GPT-5 configuration)
python -m Evaluation.MMAR_GPT5 --limit 50

# MMAU-Pro (open configuration)
python -m Evaluation.MMAUPro_Open --limit 25
```

Each runner downloads the corresponding Hugging Face dataset on first use and writes optional JSON outputs when `--output` is provided.

## Configurations

Configuration files live in `configs/` and describe the orchestrator plus the set of enabled tools. Duplicate the YAMLs to experiment with alternative tool suites or decoding parameters.

Key fields:

- `orchestrator.llm_type`: `openai` (GPT-5) or `vllm` (DeepSeek). Use `llm_url` and `api_key_env` to point to custom endpoints.
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

## License

This repository is released under the MIT license. Check individual model licenses before downloading or serving them locally.
