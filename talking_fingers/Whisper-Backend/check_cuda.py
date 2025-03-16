import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check CUDA availability
if torch.cuda.is_available():
    print("CUDA is available. GPU is detected.")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available. Using CPU instead.")
