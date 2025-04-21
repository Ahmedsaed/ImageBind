#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.quantization
from collections import defaultdict

# Import your ImageBind model
from imagebind.models.quantized_imagebind_model import (
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            module.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    # Configure quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {
            nn.Linear,
            nn.LayerNorm,
            nn.Embedding,
            nn.Dropout,
            nn.GELU,
        },  # a set of layers to dynamically quantize
        dtype=dtype,  # the target dtype for quantized weights
        inplace=True,
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
    from torch.quantization.observer import HistogramObserver
    from torch.quantization import QConfig

    q_config = QConfig(
        activation=HistogramObserver.with_args(quant_max=255, quant_min=0),
        weight=torch.quantization.default_per_channel_weight_observer,
    )

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
def get_model_size(model):
    """
    Returns the size of the model in megabytes.
    """
    import os

    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb


def create_dummy_data(batch_size=10):
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
                ModalityType.VISION: torch.randn(3, 224, 224),
                ModalityType.TEXT: torch.randint(0, 49408, (77,)),
                ModalityType.AUDIO: torch.randn(3, 1, 128, 204),
                ModalityType.DEPTH: torch.randn(1, 224, 224),
                ModalityType.THERMAL: torch.randn(1, 224, 224),
                ModalityType.IMU: torch.randn(6, 2000),
            }
            return dummy_data

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


# ----------------------------------------
# Benchmarking Functions
# ----------------------------------------
def benchmark_model_speed(model, data_loader, num_runs=50, num_samples=10):
    """
    Benchmark model inference speed.

    Args:
        model: The model to benchmark
        data_loader: DataLoader with dummy data
        num_runs: Number of runs for benchmarking
        num_samples: Number of samples to process during benchmarking

    Returns:
        Average inference time in milliseconds
    """
    import time

    with torch.no_grad():
        # Warm-up
        for _ in range(1):
            model(next(iter(data_loader)))

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                for btch_idx, sample in enumerate(data_loader):
                    if btch_idx >= num_samples:
                        break
                    model(sample)

        end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms


def compare_outputs(original_outputs, quantized_outputs):
    """
    Compare the outputs of the original and quantized models.

    Args:
        original_outputs: Outputs from the original model
        quantized_outputs: Outputs from the quantized model

    Returns:
        A dictionary with average similarity scores for each modality
    """
    modality_scores = {}

    for i in range(len(original_outputs)):
        original_output = original_outputs[i]
        quantized_output = quantized_outputs[i]

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

    return modality_scores


# ----------------------------------------
# Experiments
# ----------------------------------------
def test_model(model, data_loader, num_samples=5):
    """
    Test the model with dummy data
    """
    outputs = []

    with torch.no_grad():
        for btch_idx, sample in enumerate(data_loader):
            if btch_idx >= num_samples:
                break
            output = model(sample)
            outputs.append(output)

    return outputs


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the original model for comparison
    original_model = imagebind_huge(pretrained=True)
    original_model.eval()

    data_loader = create_dummy_data()

    dummy_inputs = [next(iter(data_loader)) for _ in range(1)]

    print("=== Computing Original Model Outputs ===")
    original_outputs = test_model(original_model, data_loader)
    original_model_size = get_model_size(original_model)

    print("=== Benchmarking Original Model Speed ===")
    original_time = benchmark_model_speed(
        original_model, data_loader, num_runs=5, num_samples=1
    )
    print(f"Original model inference time: {original_time:.2f} ms")

    print("=== Dynamic Quantization Int8 ===")
    quantized_model = apply_dynamic_quantization(original_model, dtype=torch.qint8)

    print("=== Static Quantization ===")
    quantized_model = apply_static_quantization(original_model, data_loader)

    print("=== Compute Quantized Model Outputs ===")
    quantized_outputs = test_model(quantized_model, data_loader)
    quantized_model_size = get_model_size(quantized_model)

    print("=== Benchmarking Quantized Model Speed ===")
    quantized_time = benchmark_model_speed(
        quantized_model, data_loader, num_runs=5, num_samples=1
    )
    print(f"Quantized model inference time: {quantized_time:.2f} ms")

    print("=== Comparing Outputs ===")
    print(f"Original model size: {original_model_size:.2f} MB")
    print(f"Quantized model size: {quantized_model_size:.2f} MB")
    print(
        f"Size reduction: {100 * (1 - (quantized_model_size / original_model_size)):.2f}%"
    )
    print(f"Speed improvement: {100 * (1 - (quantized_time / original_time)):.2f}x")
    compare_outputs(original_outputs, quantized_outputs)

    print("=== Saving Quantized Model === ")
    torch.save(
        quantized_model.state_dict(),
        ".checkpoints/imagebind_quantized.pth",
    )
