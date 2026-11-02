
import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Load MNIST dataset
train_path = r"D:\FYP-PROJECT\MNIST_CNN_FPGA\dataset\mnist_train.csv"
test_path  = r"D:\FYP-PROJECT\MNIST_CNN_FPGA\dataset\mnist_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y_train = train_df.iloc[:, 0].values
x_train = train_df.iloc[:, 1:].values

y_test = test_df.iloc[:, 0].values
x_test = test_df.iloc[:, 1:].values

# 2. Preprocess dataset
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 3. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Conv1
    tf.keras.layers.Conv2D(6, (3, 3), padding="valid"),
    tf.keras.layers.ReLU(),

    # Pool1
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    # Conv2
    tf.keras.layers.Conv2D(16, (3, 3), padding="valid"),
    tf.keras.layers.ReLU(),

    # Pool2
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully connected layer 1
    tf.keras.layers.Dense(64, activation="relu"),

    # Fully connected layer 2
    tf.keras.layers.Dense(10)
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 5. Display model architecture
model.summary()

# 6. Train model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 7. Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy * 100, "%")

# 8. Save trained model
model.save("digit_model.keras")

print("\nModel saved successfully as digit_model.keras")
