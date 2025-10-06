import os
from typing import List
from huggingface_hub import snapshot_download


def build_args() -> List[str]:
    model = os.getenv("MODEL", "ibm-granite/granite-speech-3.3-8b")
    port = int(os.getenv("PORT", "4000"))
    dtype = os.getenv("DTYPE", "auto")
    max_model_len = os.getenv("MAX_MODEL_LEN", "32768")
    gpu_mem = os.getenv("GPU_MEMORY_UTILIZATION", "0.45")

    # Resolve LoRA path (granite speech includes audio-specific LoRA in model dir)
    model_path = snapshot_download(model)
    lora_modules = [f"speech={model_path}"]

    # Write hostname mapping to hostnames.txt
    try:
        host = os.getenv("HOST", "127.0.0.1")
        hostname = os.popen("hostname").read().strip() or host
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        hostnames_path = os.path.join(root_dir, 'hostnames.txt')
        # Replace or insert line for GRANITE_SPEECH_SERVER
        lines = []
        if os.path.exists(hostnames_path):
            with open(hostnames_path, 'r') as f:
                lines = f.readlines()
        new_line = f"GRANITE_SPEECH_SERVER=http://{hostname}:{port}/v1\n"
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("GRANITE_SPEECH_SERVER="):
                lines[i] = new_line
                found = True
                break
        if not found:
            lines.append(new_line)
        with open(hostnames_path, 'w') as f:
            f.writelines(lines)
    except Exception:
        pass

    bind_host = os.getenv("BIND_HOST", "0.0.0.0")

    args = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--host",
        bind_host,
        "--trust-remote-code",
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--max-num-seqs",
        "2",
        "--enable-lora",
        "--max-lora-rank",
        "64",
    ]
    for lm in lora_modules:
        args += ["--lora-modules", lm]
    return args


def main():
    args = build_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()


