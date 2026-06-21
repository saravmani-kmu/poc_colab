"""Push the TTS kernel to Kaggle and trigger execution."""

import subprocess
import sys
import time
import os

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kaggle_kernel")
KERNEL_SLUG = "saravmani/poc-tamil-tts-indic-parler"


def run_cmd(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def push():
    print("=== Pushing kernel to Kaggle ===")
    code = run_cmd(f'kaggle kernels push -p "{KERNEL_DIR}"')
    if code != 0:
        print("ERROR: Failed to push kernel")
        sys.exit(1)
    print("\nKernel pushed and running on Kaggle!")


def check_status():
    print("\n=== Checking kernel status ===")
    run_cmd(f"kaggle kernels status {KERNEL_SLUG}")


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
    output_dir = os.path.join(os.path.dirname(__file__), "kaggle_output")
    os.makedirs(output_dir, exist_ok=True)
    run_cmd(f'kaggle kernels output {KERNEL_SLUG} -p "{output_dir}"')
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    push()
    success = poll_until_complete()
    if success:
        download_output()
