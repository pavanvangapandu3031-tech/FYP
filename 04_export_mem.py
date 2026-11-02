
import os
import numpy as np

# ==========================================
# 1. Define folders
# ==========================================

input_folder = "lenet_weights/int8"
output_folder = "lenet_weights/mem"

# ==========================================
# 2. Create memory file directory
# ==========================================

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 3. Define CNN layers
# ==========================================

layers = [
    "conv1",
    "conv2",
    "fc1",
    "fc2"
]

# ==========================================
# 4. Export INT8 weights to .mem files
# ==========================================

for layer in layers:

    # Load INT8 weights
    weights = np.load(
        f"{input_folder}/{layer}_weights_int8.npy"
    )

    # Flatten weights
    flat_weights = weights.flatten()

    # Create memory file
    output_path = f"{output_folder}/{layer}_weights.mem"

    with open(output_path, "w") as f:

        for value in flat_weights:

            # Convert signed INT8 to hexadecimal byte
            hex_value = int(value) & 0xFF

            f.write(f"{hex_value:02X}\n")

    print(f"{layer}: {len(flat_weights)} weights exported")

# ==========================================
# 5. Completion message
# ==========================================

print("\nAll INT8 weight memory files created!")
