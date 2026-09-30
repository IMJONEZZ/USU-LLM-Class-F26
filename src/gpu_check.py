import subprocess
import time

import modal

app = modal.App("gpu-check")


@app.function(gpu="T4")
def check_gpu():
    subprocess.run(["nvidia-smi"], check=True)
    # this gives me time to open up a new terminal and check to see if the gpu is actually running
    time.sleep(30)


@app.local_entrypoint()
def main():
    check_gpu.remote()
