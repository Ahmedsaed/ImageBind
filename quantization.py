#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.quantization
from collections import defaultdict

# Import your ImageBind model
from imagebind.models.imagebind_model import (
    imagebind_huge,
    ModalityType,
)


# ----------------------------------------
# Approach 1: Dynamic Quantization
# ----------------------------------------
def apply_dynamic_quantization(model, dtype=torch.qint8):
    """
    Dynamic quantization - quantizes the weights of linear and conv layers
    to int8 and performs calculations using int8.
    Dynamically quantizes activations during inference.
    """
    print("Applying dynamic quantization...")

    # Configure quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # the original model
        {nn.Linear},  # a set of layers to dynamically quantize
        dtype=dtype,  # the target dtype for quantized weights
    )

    return quantized_model


# ----------------------------------------
# Approach 2: Static Quantization
# ----------------------------------------
def prepare_for_static_quantization(model):
    """
    Prepares the model for static quantization by adding observers.
    """
    from torch.quantization import QuantStub, DeQuantStub

    # Set the model to evaluation mode
    model.eval()

    # Set the backend and specify which layers to quantize
    model.qconfig = torch.quantization.get_default_qconfig("x86")

    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            mod.qconfig = torch.quantization.float_qparams_weight_only_qconfig
        if isinstance(mod, (nn.Conv3d, nn.LayerNorm)):
            mod.qconfig = None

    # Create quantization stubs for each modality
    model.quant_stubs = nn.ModuleDict()
    for modality in vars(ModalityType).values():
        model.quant_stubs[modality] = QuantStub()
    model.dequant = DeQuantStub()

    original_forward = model.forward

    def wrapped_forward(inputs_dict):
        # Quantize each modality input
        quantized_inputs = {}
        for modality, tensor in inputs_dict.items():
            if modality in model.quant_stubs:
                quantized_inputs[modality] = model.quant_stubs[modality](tensor)
            else:
                quantized_inputs[modality] = tensor

        # Process with original forward
        outputs = original_forward(quantized_inputs)

        # Dequantize outputs
        if isinstance(outputs, dict):
            dequantized_outputs = {}
            for key, tensor in outputs.items():
                dequantized_outputs[key] = model.dequant(tensor)
            return dequantized_outputs
        else:
            return model.dequant(outputs)

    model.forward = wrapped_forward

    # Prepare the model for static quantization
    prepared_model = torch.quantization.prepare(model)

    return prepared_model


def calibrate_model(prepared_model, calibration_data_loader):
    """
    Runs the model on calibration data to collect statistics for quantization.
    """
    # Calibration
    print("Calibrating model...")
    with torch.no_grad():
        for batch_idx, sample in enumerate(calibration_data_loader):
            # Forward pass for calibration
            prepared_model(sample)

            # Limit calibration to a few batches for speed
            if batch_idx >= 10:
                break


def convert_to_static_quantized(prepared_model):
    """
    Converts the prepared model to a statically quantized model.
    """
    # Convert the model to a quantized model
    quantized_model = torch.quantization.convert(prepared_model)

    return quantized_model


def apply_static_quantization(model, calibration_data):
    """
    Quantizes the ImageBind model using either dynamic or static quantization.

    Args:
        use_static: If True, use static quantization. Otherwise, use dynamic quantization.

    Returns:
        A quantized version of the model.
    """
    print("Applying static quantization...")

    # Prepare for static quantization
    model = prepare_for_static_quantization(model)

    # Calibrate
    calibrate_model(model, calibration_data)

    # Convert to quantized model
    model = convert_to_static_quantized(model)

    return model


# ----------------------------------------
# Model Loaders
# ----------------------------------------
def load_dynamic_quantized_model(model_path, device="cpu"):
    """
    Load a dynamically quantized ImageBind model from a saved state dict.

    Args:
        model_path (str): Path to the saved dynamically quantized model weights
        device (str): Device to load the model on ('cpu', 'cuda')

    Returns:
        torch.nn.Module: The loaded dynamically quantized model
    """
    print(f"Loading dynamically quantized model from {model_path}")

    # First, create the base model with the same architecture
    model = imagebind_huge(pretrained=False)
    model.eval()  # Important to set to evaluation mode

    # Apply dynamic quantization to the model (same as during saving)
    quantized_model = apply_dynamic_quantization(model)

    # Load the state dictionary from file
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    quantized_model.load_state_dict(state_dict)

    return quantized_model.to(device)


def load_static_quantized_model(model_path, device="cpu"):
    """
    Load a statically quantized ImageBind model from a saved state dict.

    Args:
        model_path (str): Path to the saved statically quantized model weights
        device (str): Device to load the model on (should be 'cpu' for static quantization)

    Returns:
        torch.nn.Module: The loaded statically quantized model
    """
    if device != "cpu":
        print(
            "Warning: Static quantization was done for CPU inference. Forcing device to 'cpu'."
        )
        device = "cpu"

    print(f"Loading statically quantized model from {model_path}")

    # Create the base model with the same architecture
    model = imagebind_huge(pretrained=False)
    model.eval()  # Important to set to evaluation mode

    # Prepare the model for static quantization (same as during saving)
    prepared_model = prepare_for_static_quantization(model)

    # Convert the model to a statically quantized model
    quantized_model = convert_to_static_quantized(prepared_model)

    # Load the state dictionary from file
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    quantized_model.load_state_dict(state_dict)

    return quantized_model.to(device)


# ----------------------------------------
# Helper Functions
# ----------------------------------------
def print_used_layers_with_count(model):
    layer_counts = defaultdict(int)

    def _explore(module):
        for child in module.children():
            if isinstance(child, nn.Module):
                layer_type = type(child)
                if layer_type.__module__.startswith("torch.nn"):
                    layer_counts[layer_type.__name__] += 1
                _explore(child)

    _explore(model)

    print("Used nn layers (with count):")
    for layer_name in sorted(layer_counts):
        print(f"- {layer_name}: {layer_counts[layer_name]}")


def summarize_dtypes(model):
    dtype_count = defaultdict(int)
    for name, param in model.named_parameters():
        dtype_count[str(param.dtype)] += 1
    return dict(dtype_count)


def get_model_size(model):
    """
    Returns the size of the model in megabytes.
    """
    import os

    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb


def create_dummy_data():
    """
    Creates a dummy dataset for calibration.

    Returns:
        A DataLoader with dummy data.
    """
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Create dummy inputs for each modality
            dummy_data = {
                ModalityType.VISION: torch.randn(3, 3, 224, 224),
                ModalityType.TEXT: torch.randint(0, 49408, (77,)),
                ModalityType.AUDIO: torch.randn(1, 128, 204),
                # ModalityType.THERMAL: torch.randn(1, 1, 2, 224, 224),
                # ModalityType.DEPTH: torch.randn(1, 1, 2, 224, 224),
                # ModalityType.IMU: torch.randn(1, 1, 2, 224, 224),
            }
            return dummy_data

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader


# ----------------------------------------
# Benchmarking Functions
# ----------------------------------------
def benchmark_model_speed(model, input_data, num_runs=50):
    """
    Benchmark model inference speed.

    Args:
        model: The model to benchmark
        input_data: Input data for the model
        num_runs: Number of runs for benchmarking

    Returns:
        Average inference time in milliseconds
    """
    import time

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(input_data)

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_data)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms


def compare_models(original_model, quantized_model, data_loader, num_samples=10):
    """
    Compare the original and quantized models in terms of:
    1. Size
    2. Performance (speed)
    3. Accuracy (output similarity)
    """
    # Create dummy input
    dummy_inputs = [next(iter(data_loader)) for _ in range(num_samples)]

    # 1. Compare size
    # original_size = get_model_size(original_model)
    # quantized_size = get_model_size(quantized_model)

    # print(f"Original model size: {original_size:.2f} MB")
    # print(f"Quantized model size: {quantized_size:.2f} MB")
    # print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")

    # 2. Compare performance
    original_model.eval()
    quantized_model.eval()

    # quantized_time = benchmark_model_speed(quantized_model, dummy_inputs[0])
    # original_time = benchmark_model_speed(original_model, dummy_inputs[0])

    # print(f"Original model inference time: {original_time:.2f} ms")
    # print(f"Quantized model inference time: {quantized_time:.2f} ms")
    # print(f"Speed improvement: {(1 - quantized_time/original_time)*100:.2f}%")

    # 3. Compare output (accuracy) across multiple samples
    modality_scores = {}

    with torch.no_grad():
        for sample in dummy_inputs:
            original_output = original_model(sample)
            quantized_output = quantized_model(sample)

            for modality in original_output:
                if modality in quantized_output:
                    original_out = original_output[modality]
                    quantized_out = quantized_output[modality]

                    similarity = torch.nn.functional.cosine_similarity(
                        original_out.view(1, -1), quantized_out.view(1, -1)
                    ).item()

                    modality_scores.setdefault(modality, []).append(similarity)

    for modality, scores in modality_scores.items():
        avg_similarity = sum(scores) / len(scores)
        print(f"Average output similarity for {modality}: {avg_similarity:.6f}")


# ----------------------------------------
# Experiments
# ----------------------------------------
def dynamic_quantization_int8_example(original_model, data_loader):
    print("=== Dynamic Quantization Int8 ===")
    dynamic_quantized_model = apply_dynamic_quantization(
        original_model, dtype=torch.qint8
    )

    print("=== Comparison: Original vs Dynamic Quantization Int8 ===")
    compare_models(original_model, dynamic_quantized_model, data_loader)

    # Save the quantized models
    torch.save(
        dynamic_quantized_model.state_dict(),
        ".checkpoints/imagebind_dynamic_quantized_int8.pth",
    )


def static_quantization_example(original_model, data_loader):
    print("=== Static Quantization ===")
    static_quantized_model = apply_static_quantization(original_model, data_loader)

    # Save the quantized models
    torch.save(
        static_quantized_model.state_dict(),
        ".checkpoints/imagebind_static_quantized.pth",
    )

    # load model
    model = imagebind_huge(pretrained=False)

    model = prepare_for_static_quantization(model)

    model = convert_to_static_quantized(model)

    # Load the state dictionary from file
    state_dict = torch.load(
        ".checkpoints/imagebind_static_quantized.pth",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(state_dict)

    print("=== Comparison: Original vs Static Quantization ===")
    compare_models(original_model, model, data_loader)


def load_dynamic_quantized_model_example(original_model, quantized_model_path):
    quantized_model = load_dynamic_quantized_model(quantized_model_path)

    print("=== Comparison: Original vs Loaded Dynamic Quantized Model ===")
    compare_models(original_model, quantized_model, data_loader)


def load_static_quantized_model_example(original_model, quantized_model_path):
    quantized_model = load_static_quantized_model(quantized_model_path)

    print("=== Comparison: Original vs Loaded Static Quantized Model ===")
    compare_models(original_model, quantized_model, data_loader)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the original model for comparison
    original_model = imagebind_huge(pretrained=True)
    original_model.eval()

    data_loader = create_dummy_data()

    # dynamic_quantization_int8_example(original_model, data_loader)

    static_quantization_example(original_model, data_loader)

    # Load the quantized models for comparison
    # load_dynamic_quantized_model_example(
    #     original_model, ".checkpoints/imagebind_dynamic_quantized_int8.pth"
    # )

    # load_static_quantized_model_example(
    #     original_model, ".checkpoints/imagebind_static_quantized.pth"
    # )
