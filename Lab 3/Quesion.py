import numpy as np

def initialize_weights(input_size, hidden_sizes, output_size):
    weights = []

    weights.append(np.random.randn(input_size, hidden_sizes[0]))

    for i in range(len(hidden_sizes) - 1):
        weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i + 1]))

    weights.append(np.random.randn(hidden_sizes[-1], output_size))
    
    return weights

def forward(x, weights):
    a = [x]
    z = []

    for i, w in enumerate(weights):
        z_i = np.dot(a[-1], w)
        z.append(z_i)
        
        if i < len(weights) - 1:  
            a_i = sigmoid(z_i)  
        else: 
            a_i = relu(z_i) 

        a.append(a_i)

    return a, z

def backpropagate(x, y, a, z, weights, learning_rate=0.1):
    m = y.shape[0]
    delta = a[-1] - y  

    for i in reversed(range(len(weights))):
        layer_input = a[i]
        gradient = np.dot(layer_input.T, delta) / m
        weights[i] -= learning_rate * gradient
        
        if i > 0: 
            delta = np.dot(delta, weights[i].T) * derivative(z[i - 1], i)

    return weights

def train(x, y, hidden_sizes, epochs=10000, learning_rate=0.1):
    input_size = x.shape[1]
    output_size = y.shape[1]
    
    weights = initialize_weights(input_size, hidden_sizes, output_size)

    for _ in range(epochs):
        a, z = forward(x, weights)
        weights = backpropagate(x, y, a, z, weights, learning_rate)

    return weights

def predict(x, weights):
    a, _ = forward(x, weights)
    return (a[-1] > 0).astype(int) 

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivative(z, layer_index):
    if layer_index % 2 == 0: 
        return 1 - np.tanh(z) ** 2
    else:  
        return (z > 0).astype(float)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])


hidden_sizes = [2] 
weights = train(X, y, hidden_sizes, epochs=1000, learning_rate=0.01)

predictions = predict(X, weights)
print("Predictions:", predictions.flatten())
print("Weights: ",weights[0])
