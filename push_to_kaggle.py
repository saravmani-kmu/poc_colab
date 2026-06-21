"""Push the TTS kernel to Kaggle and trigger execution."""

import subprocess
import sys
import time
import os
import json
import copy

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_kernel")
KERNEL_SLUG = "saravmani/poc-tamil-tts-indic-parler"
NOTEBOOK_PATH = os.path.join(KERNEL_DIR, "kaggle_notebook.ipynb")

HF_TOKEN_FILE = r"D:\project_kyog\secrets\tokens.txt"


def get_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    if os.path.exists(HF_TOKEN_FILE):
        with open(HF_TOKEN_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1]
                if line.startswith("hf_"):
                    return line
    print("ERROR: HF_TOKEN not found. Set HF_TOKEN env var or add to secrets/tokens.txt")
    sys.exit(1)


def inject_token_and_push(hf_token):
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    nb_patched = copy.deepcopy(nb)
    for cell in nb_patched["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "HF_TOKEN" in src and "login" in src:
                cell["source"] = [
                    f'import os\n',
                    f'os.environ["HF_TOKEN"] = "{hf_token}"\n',
                    f'from huggingface_hub import login\n',
                    f'login(token=os.environ["HF_TOKEN"])\n',
                    f'print("Logged in to HuggingFace")\n',
                ]
                break

    patched_path = NOTEBOOK_PATH + ".tmp"
    with open(patched_path, "w") as f:
        json.dump(nb_patched, f, indent=2)

    os.replace(patched_path, NOTEBOOK_PATH)
    print("=== Pushing kernel to Kaggle ===")
    code = run_cmd(f'kaggle kernels push -p "{KERNEL_DIR}"')

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)

    if code != 0:
        print("ERROR: Failed to push kernel")
        sys.exit(1)
    print("\nKernel pushed and running on Kaggle!")


def run_cmd(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def poll_until_complete(max_wait=1800, interval=30):
    print(f"\n=== Polling status every {interval}s (max {max_wait}s) ===")
    elapsed = 0
    while elapsed < max_wait:
        result = subprocess.run(
            f"kaggle kernels status {KERNEL_SLUG}",
            shell=True, capture_output=True, text=True,
        )
        status = result.stdout.strip()
        print(f"[{elapsed}s] {status}")

        if "complete" in status.lower():
            print("\nKernel completed!")
            return True
        if "error" in status.lower() or "cancelAcknowledged" in status.lower():
            print("\nKernel failed!")
            return False

        time.sleep(interval)
        elapsed += interval

    print("\nTimeout waiting for kernel")
    return False


def download_output():
    print("\n=== Downloading output ===")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_output")
    os.makedirs(output_dir, exist_ok=True)
    run_cmd(f'kaggle kernels output {KERNEL_SLUG} -p "{output_dir}"')
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    hf_token = get_hf_token()
    inject_token_and_push(hf_token)
    success = poll_until_complete()
    if success:
        download_output()
