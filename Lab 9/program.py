import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("TCS.NS.csv")  
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 50
X, y = create_sequences(prices_scaled, sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=False, input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

import matplotlib.pyplot as plt
plt.plot(true_prices, label="True Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.legend()
plt.show()
