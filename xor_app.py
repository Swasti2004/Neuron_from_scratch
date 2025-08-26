import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Helper functions
# -----------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_xor(lr=0.1, epochs=10000):
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    np.random.seed(42)
    input_neurons = 2
    hidden_neurons = 2
    output_neurons = 1

    W1 = np.random.randn(input_neurons, hidden_neurons)
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, output_neurons)
    b2 = np.zeros((1, output_neurons))

    losses = []

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        output = sigmoid(z2)

        # Loss (MSE)
        loss = np.mean((y - output) ** 2)
        losses.append(loss)

        # Backpropagation
        d_output = (output - y) * sigmoid_derivative(output)
        d_W2 = np.dot(a1.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2

    return W1, b1, W2, b2, losses

def predict(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    return sigmoid(z2)

def plot_decision_boundary(W1, b1, W2, b2):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(grid, W1, b1, W2, b2)
    preds = preds.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, levels=[-0.1, 0.5, 1.1], cmap=plt.cm.Paired, alpha=0.6)
    ax.scatter([0,0,1,1], [0,1,0,1], c=[0,1,1,0], edgecolors='k', cmap=plt.cm.Paired, s=80)
    ax.set_title("XOR Decision Boundary")
    return fig

# -----------------------
# Streamlit App
# -----------------------

st.title("XOR Neural Network from Scratch")
st.write("A simple 2-layer NN trained with sigmoid activation to learn XOR.")

# Sidebar inputs
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 1000, 20000, 10000, 1000)

if st.button("Train Model"):
    W1, b1, W2, b2, losses = train_xor(lr=learning_rate, epochs=epochs)

    # Show final predictions
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    preds = predict(X, W1, b1, W2, b2)
    st.subheader("Predictions")
    for inp, p in zip(X, preds):
        st.write(f"Input: {inp}, Predicted: {np.round(p[0], 3)}")

    # Loss curve
    fig_loss, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    st.pyplot(fig_loss)

    # Decision boundary
    fig_boundary = plot_decision_boundary(W1, b1, W2, b2)
    st.pyplot(fig_boundary)

    st.success("Training complete. XOR learned!")
else:
    st.info("Click 'Train Model' to start training.")
