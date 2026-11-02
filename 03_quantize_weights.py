
import os
import numpy as np

# ==========================================
# 1. Create INT8 storage directory
# ==========================================

os.makedirs("lenet_weights/int8", exist_ok=True)

# ==========================================
# 2. INT8 quantization function
# ==========================================

def quantize_int8(weights):

    max_value = np.max(np.abs(weights))

    scale = max_value / 127.0

    quantized = np.round(weights / scale)

    quantized = np.clip(quantized, -127, 127)

    quantized = quantized.astype(np.int8)

    return quantized, scale

# ==========================================
# 3. Define CNN layers
# ==========================================

layers = [
    "conv1",
    "conv2",
    "fc1",
    "fc2"
]

scales = {}

# ==========================================
# 4. Quantize weights layer by layer
# ==========================================

for layer in layers:

    # Load original FP32 weights
    weights = np.load(
        f"lenet_weights/weights/{layer}_weights.npy"
    )

    # Quantize weights
    quantized_weights, scale = quantize_int8(weights)

    # Save INT8 weights
    np.save(
        f"lenet_weights/int8/{layer}_weights_int8.npy",
        quantized_weights
    )

    # Store scale
    scales[layer] = scale

    # Display results
    print("\nLayer:", layer)
    print("Original shape:", weights.shape)
    print("INT8 shape:", quantized_weights.shape)
    print("Original range:", weights.min(), weights.max())
    print("INT8 range:", quantized_weights.min(), quantized_weights.max())
    print("Scale:", scale)

# ==========================================
# 5. Save quantization scales
# ==========================================

np.savez(
    "lenet_weights/int8/weight_scales.npz",
    **scales
)

print("\nAll weights quantized successfully!")
