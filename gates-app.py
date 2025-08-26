import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def step(x): return np.where(x >= 0, 1, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))

# XOR forward pass
def forward_xor(X, W1, b1, W2, b2):
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    return output

# Plot decision boundary
def plot_boundary(predict_func, X, y):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, levels=[-0.1, 0.5, 1.1], alpha=0.6, cmap=plt.cm.Paired)
    ax.scatter(X[:,0], X[:,1], c=y.flatten(), edgecolors='k', s=80, cmap=plt.cm.Paired)
    ax.set_title("Decision Boundary")
    return fig

# Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
truth = {
    "AND": np.array([[0],[0],[0],[1]]),
    "OR":  np.array([[0],[1],[1],[1]]),
    "NAND":np.array([[1],[1],[1],[0]]),
    "XOR": np.array([[0],[1],[1],[0]])
}

st.title("Interactive Logic Gate Visualizer")
gate = st.sidebar.selectbox("Select Gate", ["AND","OR","NAND","XOR"])
mode = st.sidebar.radio("Mode", ["Manual weights", "Auto-train demo"])

y = truth[gate]

if gate != "XOR":
    if mode == "Manual weights":
        w1 = st.sidebar.slider("Weight w1", -5.0, 5.0, 0.0, 0.1)
        w2 = st.sidebar.slider("Weight w2", -5.0, 5.0, 0.0, 0.1)
        b = st.sidebar.slider("Bias", -5.0, 5.0, 0.0, 0.1)
        def predict_func(inp): return step(np.dot(inp, [w1,w2]) + b)
    else:
        # quick perceptron training
        lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
        epochs = st.sidebar.slider("Epochs", 10, 1000, 100)
        w = np.zeros(2); bias = 0
        for _ in range(epochs):
            for xi, target in zip(X, y):
                out = step(np.dot(xi, w) + bias)
                err = target - out
                w += lr * err * xi
                bias += lr * err
        w1,w2 = w; b=bias
        def predict_func(inp): return step(np.dot(inp, [w1,w2]) + b)

    preds = predict_func(X)
    st.write("Predictions:", preds.flatten().tolist())
    st.write(f"Accuracy: {np.mean(preds==y.flatten())*100:.2f}%")
    st.pyplot(plot_boundary(predict_func, X, y))

else:
    if mode == "Manual weights":
        # Hidden layer weights (2x2) and bias (1x2)
        h_w11 = st.sidebar.slider("Hidden w11", -10.0, 10.0, 1.0, 0.1)
        h_w12 = st.sidebar.slider("Hidden w12", -10.0, 10.0, 1.0, 0.1)
        h_w21 = st.sidebar.slider("Hidden w21", -10.0, 10.0, 1.0, 0.1)
        h_w22 = st.sidebar.slider("Hidden w22", -10.0, 10.0, 1.0, 0.1)
        h_b1 = st.sidebar.slider("Hidden bias1", -10.0, 10.0, 0.0, 0.1)
        h_b2 = st.sidebar.slider("Hidden bias2", -10.0, 10.0, 0.0, 0.1)
        o_w1 = st.sidebar.slider("Out w1", -10.0, 10.0, 1.0, 0.1)
        o_w2 = st.sidebar.slider("Out w2", -10.0, 10.0, 1.0, 0.1)
        o_b = st.sidebar.slider("Out bias", -10.0, 10.0, 0.0, 0.1)

        W1 = np.array([[h_w11,h_w12],[h_w21,h_w22]])
        b1 = np.array([[h_b1,h_b2]])
        W2 = np.array([[o_w1],[o_w2]])
        b2 = np.array([[o_b]])
    else:
        # quick backprop training
        lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
        epochs = st.sidebar.slider("Epochs", 100, 20000, 10000)
        np.random.seed(42)
        W1 = np.random.randn(2,2); b1=np.zeros((1,2))
        W2 = np.random.randn(2,1); b2=np.zeros((1,1))
        for _ in range(epochs):
            hidden = sigmoid(np.dot(X,W1)+b1)
            out = sigmoid(np.dot(hidden,W2)+b2)
            d_out = (out-y)*out*(1-out)
            d_W2 = np.dot(hidden.T,d_out); d_b2=np.sum(d_out,axis=0,keepdims=True)
            d_hidden = np.dot(d_out,W2.T)*hidden*(1-hidden)
            d_W1 = np.dot(X.T,d_hidden); d_b1=np.sum(d_hidden,axis=0,keepdims=True)
            W2 -= lr*d_W2; b2-=lr*d_b2; W1-=lr*d_W1; b1-=lr*d_b1

    def predict_func(inp): return np.round(forward_xor(inp,W1,b1,W2,b2))
    preds = predict_func(X)
    st.write("Predictions:", preds.flatten().tolist())
    st.write(f"Accuracy: {np.mean(preds==y.flatten())*100:.2f}%")
    st.pyplot(plot_boundary(predict_func, X, y))
