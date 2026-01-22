import torch
import time
import logging
import numpy as np
from copy import deepcopy
from resnet import QuoridorNet

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def benchmark_inference(model, input_tensor, precision_name, num_warmup=50, num_iters=200):
    """
    Runs inference loop and returns (avg_latency_ms, samples_per_sec, output_p, output_v)
    """
    # Create CUDA Events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Benchmark
    torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        for _ in range(num_iters):
            p, v = model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / num_iters
    batch_size = input_tensor.shape[0]
    samples_per_sec = (batch_size * num_iters) / (total_time_ms / 1000.0)

    return avg_latency_ms, samples_per_sec, p, v

def compare_outputs(baseline_p, baseline_v, test_p, test_v):
    """
    Calculates MSE and Max Absolute Error between baseline (FP32) and test outputs.
    """
    # Ensure inputs are FP32 for fair comparison
    base_p = baseline_p.float()
    base_v = baseline_v.float()
    test_p = test_p.float()
    test_v = test_v.float()

    p_mse = torch.nn.functional.mse_loss(base_p, test_p).item()
    v_mse = torch.nn.functional.mse_loss(base_v, test_v).item()

    p_max_diff = (base_p - test_p).abs().max().item()
    v_max_diff = (base_v - test_v).abs().max().item()

    return p_mse, v_mse, p_max_diff, v_max_diff

def main():
    # --- Configuration ---
    BATCH_SIZE = 128  # Large enough to saturate the GPU
    CHANNELS = 128
    BLOCKS = 15
    DEVICE = "cuda"

    if not torch.cuda.is_available():
        logging.error("CUDA not found! This script requires a GPU.")
        return

    logging.info(f"Benchmarking on: {torch.cuda.get_device_name(0)}")

    # 1. Setup Baseline Model (FP32)
    model = QuoridorNet(num_channels=CHANNELS, num_blocks=BLOCKS).to(DEVICE)
    model.eval()

    # Generate random dummy input (Batch, 6, 9, 9)
    # We use random numbers, but seed them so it's deterministic across runs if needed
    torch.manual_seed(42)
    dummy_input = torch.randn(BATCH_SIZE, 6, 9, 9, device=DEVICE)

    logging.info("-" * 80)
    logging.info(f"Mode      | Latency (ms) | Throughput (img/s) | Speedup | P_MSE (Error) | V_MSE (Error)")
    logging.info("-" * 80)

    # --- Run FP32 Baseline ---
    lat_32, tpt_32, p_32, v_32 = benchmark_inference(model, dummy_input, "FP32")
    logging.info(f"FP32      | {lat_32:12.4f} | {tpt_32:18.2f} | 1.00x   | N/A           | N/A")

    # --- Run FP16 (Half Precision) ---
    # We cast the model and input to FP16
    model_fp16 = deepcopy(model).half()
    input_fp16 = dummy_input.half()

    lat_16, tpt_16, p_16, v_16 = benchmark_inference(model_fp16, input_fp16, "FP16")
    p_mse, v_mse, _, _ = compare_outputs(p_32, v_32, p_16, v_16)
    logging.info(f"FP16      | {lat_16:12.4f} | {tpt_16:18.2f} | {tpt_16/tpt_32:.2f}x   | {p_mse:.2e}      | {v_mse:.2e}")

    # --- Run BF16 (BFloat16) ---
    # Blackwell/Hopper GPUs love BF16. It has same range as FP32, less precision.
    model_bf16 = deepcopy(model).bfloat16()
    input_bf16 = dummy_input.bfloat16()

    lat_bf16, tpt_bf16, p_bf16, v_bf16 = benchmark_inference(model_bf16, input_bf16, "BF16")
    p_mse, v_mse, _, _ = compare_outputs(p_32, v_32, p_bf16, v_bf16)
    logging.info(f"BF16      | {lat_bf16:12.4f} | {tpt_bf16:18.2f} | {tpt_bf16/tpt_32:.2f}x   | {p_mse:.2e}      | {v_mse:.2e}")

    # --- Run FP8 (Experimental / Blackwell Specific) ---
    # PyTorch 2.2+ supports float8_e4m3fn (better for inference) and float8_e5m2 (better for training)
    # Note: Native casting .to(torch.float8) is limited for nn.Modules. 
    # We usually use torch.compile or TE. For this script, we check if we can cast input/weights.

    try:
        # Check for FP8 support
        fp8_dtype = torch.float8_e4m3fn

        # NOTE: Standard nn.Conv2d doesn't natively support FP8 inputs in eager mode easily 
        # without external libraries like TransformerEngine, but we can try a "simulated" test 
        # or use torch.amp if available.
        # For this benchmark, let's use torch.amp.autocast with 'float8' if supported (unlikely in vanilla Torch yet)
        # Instead, we will try to cast tensors manually to check support.

        # Since strict FP8 inference usually requires specialized kernels (TransformerEngine),
        # we will skip the "Cast Model" approach which would crash on standard Conv2d
        # and instead just check tensor creation to confirm the card supports it.

        test_tensor = torch.zeros(10, device=DEVICE, dtype=fp8_dtype)
        logging.info("-" * 80)
        logging.info("FP8 Support Detected (torch.float8_e4m3fn)!")
        logging.info("To run true FP8 inference, use 'import transformer_engine.pytorch as te'")
        logging.info("and replace nn.Conv2d with te.Conv2d, or use 'torch.compile(..., mode=\"max-autotune\")'")

    except (AttributeError, RuntimeError):
        logging.info("-" * 80)
        logging.info("FP8 (e4m3fn) not natively accessible in this PyTorch version.")

    # --- Torch.compile (The 2026 Standard) ---
    # This usually fuses kernels and uses TensorCores more effectively than Eager mode
    logging.info("-" * 80)
    logging.info("Benchmarking torch.compile() [Default Mode]...")

    # We use the BF16 model for compile as it's the standard for modern AI
    model_compiled = torch.compile(model_bf16)

    # Run once to trigger compilation (this will be slow)
    _ = model_compiled(input_bf16)

    lat_c, tpt_c, p_c, v_c = benchmark_inference(model_compiled, input_bf16, "Compile")
    p_mse, v_mse, _, _ = compare_outputs(p_32, v_32, p_c, v_c) # Compare vs FP32 baseline
    logging.info(f"Compile   | {lat_c:12.4f} | {tpt_c:18.2f} | {tpt_c/tpt_32:.2f}x   | {p_mse:.2e}      | {v_mse:.2e}")

    logging.info("-" * 80)
    logging.info("Done.")

if __name__ == "__main__":
    main()
