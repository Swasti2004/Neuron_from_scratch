import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Activation functions ----------------
def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------------- Single-layer perceptron ----------------
def train_perceptron(X, y, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    losses = []
    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X, y):
            output = step(np.dot(xi, w) + b)
            error = target - output
            w += lr * error * xi
            b += lr * error
            total_error += abs(error)
        losses.append(total_error)
        if total_error == 0:
            break
    return w, b, losses

# ---------------- Multi-layer NN for XOR ----------------
def train_xor(X, y, lr=0.1, epochs=10000):
    np.random.seed(42)
    W1 = np.random.randn(2, 2)
    b1 = np.zeros((1,2))
    W2 = np.random.randn(2, 1)
    b2 = np.zeros((1,1))
    losses = []

    for _ in range(epochs):
        # forward
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        out = sigmoid(z2)
        loss = np.mean((y - out) ** 2)
        losses.append(loss)
        # backprop
        d_out = (out - y) * sigmoid_derivative(out)
        d_W2 = np.dot(a1.T, d_out)
        d_b2 = np.sum(d_out, axis=0, keepdims=True)

        d_a1 = np.dot(d_out, W2.T) * sigmoid_derivative(a1)
        d_W1 = np.dot(X.T, d_a1)
        d_b1 = np.sum(d_a1, axis=0, keepdims=True)

        # update
        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2
    return W1, b1, W2, b2, losses

def predict_xor(X, W1, b1, W2, b2):
    a1 = sigmoid(np.dot(X, W1) + b1)
    out = sigmoid(np.dot(a1, W2) + b2)
    return np.round(out)

# ---------------- Decision boundary plot ----------------
def plot_boundary_and_points(predict_func, X, y):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Paired)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired, s=80)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_title("Decision Boundary")
    return fig

# ---------------- Streamlit App ----------------
st.title("Interactive Logic Gate Trainer")

gate = st.sidebar.selectbox("Select Gate", ["AND","OR","NAND","XOR"])
lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 20000, 1000, 10)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
truth_tables = {
    "AND": np.array([0,0,0,1]),
    "OR": np.array([0,1,1,1]),
    "NAND": np.array([1,1,1,0]),
    "XOR": np.array([0,1,1,0])
}
y = truth_tables[gate].reshape(-1,1)

if st.button("Train"):
    if gate != "XOR":
        w, b, losses = train_perceptron(X, y.flatten(), lr=lr, epochs=epochs)
        predict_func = lambda inp: step(np.dot(inp, w) + b)
        preds = predict_func(X)
        acc = np.mean(preds == y.flatten()) * 100

        st.subheader(f"{gate} Gate Results")
        st.write(f"Weights: {w}, Bias: {b:.2f}")
        st.write("Predictions:", preds.tolist())
        st.write(f"Accuracy: {acc:.2f}%")

        # Plot loss
        fig_loss, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title("Total Error per Epoch")
        st.pyplot(fig_loss)

        # Plot decision boundary
        st.pyplot(plot_boundary_and_points(predict_func, X, y))

    else:
        W1,b1,W2,b2,losses = train_xor(X, y, lr=lr, epochs=epochs)
        preds = predict_xor(X, W1,b1,W2,b2)
        acc = np.mean(preds == y) * 100

        st.subheader("XOR Gate Results (2-layer NN)")
        st.write("Hidden Weights:", W1)
        st.write("Output Weights:", W2)
        st.write("Predictions:", preds.flatten().tolist())
        st.write(f"Accuracy: {acc:.2f}%")

        # Plot loss
        fig_loss, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title("Loss Curve (MSE)")
        st.pyplot(fig_loss)

        # Plot decision boundary
        predict_func = lambda inp: predict_xor(inp, W1,b1,W2,b2)
        st.pyplot(plot_boundary_and_points(predict_func, X, y))
else:
    st.info("Select parameters and click Train.")
