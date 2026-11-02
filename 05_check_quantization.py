
import numpy as np
import pandas as pd
import tensorflow as tf

# ==========================================
# 1. Load trained CNN model
# ==========================================

model = tf.keras.models.load_model("digit_model.keras")

print("Model loaded successfully!")

# ==========================================
# 2. Load quantization scales
# ==========================================

quant_data = np.load(
    "lenet_weights/int8/weight_scales.npz"
)

# ==========================================
# 3. Define CNN layer mapping
# ==========================================

layers = {
    "conv1": "conv2d",
    "conv2": "conv2d_1",
    "fc1": "dense",
    "fc2": "dense_1"
}

# ==========================================
# 4. Replace FP32 weights with
#    dequantized INT8 weights
# ==========================================

for layer_name, keras_layer_name in layers.items():

    layer = model.get_layer(keras_layer_name)

    # Load INT8 weights
    quantized_weights = np.load(
        f"lenet_weights/int8/{layer_name}_weights_int8.npy"
    )

    # Load corresponding scale
    scale = quant_data[layer_name]

    # Dequantize weights
    dequantized_weights = (
        quantized_weights.astype(np.float32) * scale
    )

    # Load original bias
    bias = np.load(
        f"lenet_weights/weights/{layer_name}_bias.npy"
    )

    # Update model weights
    layer.set_weights([
        dequantized_weights,
        bias
    ])

    print(layer_name, "weights updated successfully")

# ==========================================
# 5. Load MNIST test dataset
# ==========================================

test_path = r"D:\FYP-PROJECT\MNIST_CNN_FPGA\dataset\mnist_test.csv"

test_df = pd.read_csv(test_path)

y_test = test_df.iloc[:, 0].values
x_test = test_df.iloc[:, 1:].values

# ==========================================
# 6. Preprocess test images
# ==========================================

x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test.astype("float32") / 255.0

# ==========================================
# 7. Evaluate quantized-weight model
# ==========================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

# ==========================================
# 8. Display results
# ==========================================

print("\n===================================")
print("QUANTIZATION EVALUATION RESULTS")
print("===================================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

print("\nQuantized weight evaluation completed!")
