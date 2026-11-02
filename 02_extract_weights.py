
import os
import numpy as np
import tensorflow as tf

# 1. Load trained CNN model
model = tf.keras.models.load_model("digit_model.keras")
print("Model loaded successfully!")

# 2. Create weights storage directory
os.makedirs("lenet_weights/weights", exist_ok=True)

# 3. Extract layer weights and biases
layers = {
    "conv1": "conv2d",
    "conv2": "conv2d_1",
    "fc1": "dense",
    "fc2": "dense_1"
}

# 4. Save extracted weights

for layer_name, keras_layer_name in layers.items():

    layer = model.get_layer(keras_layer_name)

    weights, bias = layer.get_weights()

    np.save(
        f"lenet_weights/weights/{layer_name}_weights.npy",
        weights
    )

    np.save(
        f"lenet_weights/weights/{layer_name}_bias.npy",
        bias
    )

    print("\nLayer:", layer_name)
    print("Weights shape:", weights.shape)
    print("Bias shape:", bias.shape)

# 5. Completion message
print("\nAll weights and biases extracted successfully!")
