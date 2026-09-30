import modal

app = modal.App("gpu-test")

image = modal.Image.debian_slim().pip_install("torch")


@app.function(
    image=image,
    gpu="T4",
)
def test_gpu():
    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name())

    print(torch.cuda.memory_allocated())
