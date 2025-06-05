import torch

print(f"Versiune PyTorch: {torch.__version__}")
print(f"CUDA disponibil pentru PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Versiunea CUDA cu care PyTorch a fost compilat: {torch.version.cuda}")
    # ... restul printurilor pentru GPU
else:
    print("PyTorch NU a fost instalat cu suport CUDA sau există o problemă de configurare.")