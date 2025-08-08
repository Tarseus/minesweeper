import torch, os, subprocess, sys
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    try:
        print(subprocess.check_output(["nvidia-smi"]).decode().splitlines()[2])
    except Exception as e:
        print("nvidia-smi not found or no driver:", e)
