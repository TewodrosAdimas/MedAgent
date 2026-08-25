"""
Comprehensive PyTorch GPU / CUDA Verification & Benchmark Script.
Validates GPU compute, tensor operations, neural network forward/backward passes,
and memory management on the NVIDIA RTX 5050 Laptop GPU.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


def print_gpu_info():
    print("=" * 60)
    print("      PYTORCH & NVIDIA GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Built With   : {torch.version.cuda}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available on this system.")
        return False

    device_id = 0
    props = torch.cuda.get_device_properties(device_id)
    total_mem_gb = props.total_memory / (1024**3)

    print(f"Device ID         : {device_id}")
    print(f"GPU Name          : {props.name}")
    print(f"Compute Capability: sm_{props.major}{props.minor} (Major: {props.major}, Minor: {props.minor})")
    print(f"Total VRAM        : {total_mem_gb:.2f} GB ({props.total_memory:,} bytes)")
    print(f"Multi-processors  : {props.multi_processor_count}")
    print("-" * 60)
    return True


def test_gpu_neural_network():
    print("\n--- Test 1: 1D CNN Forward & Backward Pass on GPU ---")
    device = torch.device("cuda:0")

    # Define a sample 1D CNN for ECG signals
    class ECGMiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 5),  # 5 AAMI classes
            )

        def forward(self, x):
            return self.net(x)

    model = ECGMiniModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dummy batch of ECG signals: (batch_size=64, channels=1, seq_len=260)
    dummy_signals = torch.randn(64, 1, 260, device=device)
    dummy_labels = torch.randint(0, 5, (64,), device=device)

    # Forward pass
    outputs = model(dummy_signals)
    loss = criterion(outputs, dummy_labels)

    # Backward pass & Optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Input Tensor Shape   : {tuple(dummy_signals.shape)} on {dummy_signals.device}")
    print(f"Output Logits Shape  : {tuple(outputs.shape)} on {outputs.device}")
    print(f"Computed Loss        : {loss.item():.4f}")
    print(f"Allocated GPU VRAM   : {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
    print(f"Cached GPU VRAM      : {torch.cuda.memory_reserved(device) / (1024**2):.2f} MB")
    print("[PASS] 1D CNN Neural Network forward/backward pass succeeded on GPU!")


def benchmark_gpu_vs_cpu():
    print("\n--- Test 2: Matrix Multiplication Benchmark (GPU vs CPU) ---")
    size = 4000
    iterations = 5

    # CPU Benchmark
    print(f"Running {iterations}x ({size} x {size}) matrix multiplications on CPU...")
    x_cpu = torch.randn(size, size, dtype=torch.float32, device="cpu")
    start_cpu = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x_cpu, x_cpu)
    time_cpu = (time.perf_counter() - start_cpu) / iterations
    print(f"CPU Average Time     : {time_cpu * 1000:.2f} ms")

    # GPU Benchmark
    device = torch.device("cuda:0")
    print(f"Running {iterations}x ({size} x {size}) matrix multiplications on GPU...")
    x_gpu = torch.randn(size, size, dtype=torch.float32, device=device)
    
    # Warmup
    _ = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()

    start_gpu = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()
    time_gpu = (time.perf_counter() - start_gpu) / iterations
    print(f"GPU Average Time     : {time_gpu * 1000:.2f} ms")

    speedup = time_cpu / time_gpu if time_gpu > 0 else 0
    print(f"GPU Speedup Factor   : {speedup:.2f}x FASTER than CPU")
    print("[PASS] GPU Benchmark completed successfully!")


if __name__ == "__main__":
    if print_gpu_info():
        test_gpu_neural_network()
        benchmark_gpu_vs_cpu()
        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED: PyTorch is fully operational on your GPU!")
        print("=" * 60)
