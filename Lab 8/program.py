import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Build a simple model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Adversarial Training
def adversarial_training(model, x_train, y_train, epsilon=0.1, epochs=5, batch_size=64):
    def generate_adversarial_examples(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        gradients = tape.gradient(loss, x)
        adversarial_x = x + epsilon * tf.sign(gradients)
        return tf.clip_by_value(adversarial_x, 0, 1)

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = tf.convert_to_tensor(x_train[i:i + batch_size])
            y_batch = tf.convert_to_tensor(y_train[i:i + batch_size])
            adversarial_x_batch = generate_adversarial_examples(x_batch, y_batch)
            model.train_on_batch(adversarial_x_batch, y_batch)

# Tangent Propagation
def tangent_propagation(model, x_train, y_train, tangent_vectors, lambda_=0.1, epochs=5, batch_size=64):
    def tangent_loss(x, tangent_vector):
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = model(x)
        jacobian = tape.jacobian(predictions, x)
        tangent_loss = tf.reduce_sum((tf.tensordot(jacobian, tangent_vector, axes=1)) ** 2)
        return tangent_loss

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = tf.convert_to_tensor(x_train[i:i + batch_size])
            y_batch = tf.convert_to_tensor(y_train[i:i + batch_size])
            tangent_vector_batch = tangent_vectors[i:i + batch_size]

            with tf.GradientTape() as tape:
                predictions = model(x_batch)
                classification_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_batch, predictions))
                tang_loss = tf.reduce_mean([tangent_loss(x, t) for x, t in zip(x_batch, tangent_vector_batch)])
                total_loss = classification_loss + lambda_ * tang_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Tangent Classifier
def tangent_classifier(x, y, tangent_vectors):
    # Flatten data
    x_flat = x.reshape(x.shape[0], -1)
    tangent_vectors_flat = tangent_vectors.reshape(tangent_vectors.shape[0], -1)
    
    # Compute distances
    distances = np.linalg.norm(x_flat[:, None, :] - tangent_vectors_flat[None, :, :], axis=-1)
    predictions = np.argmin(distances, axis=-1)
    accuracy = np.mean(predictions == y)
    print(f"Tangent Classifier Accuracy: {accuracy * 100:.2f}%")

# Main function to demonstrate the three techniques
if __name__ == "__main__":
    model = build_model()
    
    # Generate random tangent vectors for demonstration
    tangent_vectors = np.random.normal(size=(x_train.shape[0], 28, 28))
    
    # Train with Adversarial Training
    adversarial_training(model, x_train, y_train)
    print("Adversarial Training completed")
    
    # Train with Tangent Propagation
    tangent_propagation(model, x_train, y_train, tangent_vectors)
    print("Tangent Propagation completed")
    
    # Evaluate Tangent Classifier
    tangent_classifier(x_test, np.argmax(y_test, axis=1), tangent_vectors[:len(x_test)])
