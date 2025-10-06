# see also https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_1.html
import os
from typing import List


def build_args() -> List[str]:
    model = os.getenv("MODEL", "deepseek-ai/DeepSeek-V3.1")
    port = int(os.getenv("PORT", "4005"))
    dtype = os.getenv("DTYPE", "auto")
    max_model_len = os.getenv("MAX_MODEL_LEN", "32768")

    # Write hostname mapping to hostnames.txt
    try:
        host = os.getenv("HOST", "127.0.0.1")
        hostname = os.popen("hostname").read().strip() or host
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        hostnames_path = os.path.join(root_dir, 'hostnames.txt')
        new_line = f"DEEPSEEK_SERVER=http://{hostname}:{port}/v1\n"
        lines = []
        if os.path.exists(hostnames_path):
            with open(hostnames_path, 'r') as f:
                lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith("DEEPSEEK_SERVER="):
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
        "--tensor-parallel-size",
        "8",
        "--served-model-name",
        "ds31",
        "--enable-expert-parallel",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "deepseek_v31",
        "--chat-template",
        "examples/tool_chat_template_deepseekv31.jinja",
        "--dtype",
        dtype,
        "--seed",
        "42"
    ]
    return args


def main():
    args = build_args()
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()


