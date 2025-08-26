with tab1:
    gate = st.sidebar.selectbox("Select Gate", ["AND","OR","NAND","XOR"])
    mode = st.sidebar.radio("Mode", ["Manual weights", "Auto-train demo"])
    y = truth[gate]

    if mode == "Manual weights":
        # Manual slider mode
        if gate != "XOR":
            w1 = st.sidebar.slider("Weight w1", -5.0, 5.0, 0.0, 0.1)
            w2 = st.sidebar.slider("Weight w2", -5.0, 5.0, 0.0, 0.1)
            b = st.sidebar.slider("Bias", -5.0, 5.0, 0.0, 0.1)
            predict_func = lambda inp: step(np.dot(inp, [w1,w2]) + b)
            preds = predict_func(X)
        else:
            # Manual XOR slider
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
            predict_func = lambda inp: np.round(forward_xor(inp,W1,b1,W2,b2))
            preds = predict_func(X)

        st.subheader(f"{gate} Predictions (Manual Mode)")
        st.write("Predictions:", preds.flatten().tolist())
        st.write(f"Accuracy: {np.mean(preds==y.flatten())*100:.2f}%")
        st.pyplot(plot_boundary(predict_func, X, y))

    else:
        # Auto-train mode with Train button
        lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
        epochs = st.sidebar.slider("Epochs", 10, 20000, 100)
        if st.button("Train"):
            if gate != "XOR":
                w = np.zeros(2); bias = 0
                losses = []
                for _ in range(epochs):
                    total_err = 0
                    for xi, target in zip(X, y):
                        out = step(np.dot(xi, w) + bias)
                        err = target - out
                        w += lr * err * xi
                        bias += lr * err
                        total_err += abs(err)
                    losses.append(total_err)
                    if total_err == 0: break
                predict_func = lambda inp: step(np.dot(inp, w) + bias)
                preds = predict_func(X)
                st.subheader(f"{gate} Predictions (Trained)")
                st.write("Weights:", w, "Bias:", bias)
                st.write("Predictions:", preds.flatten().tolist())
                st.write(f"Accuracy: {np.mean(preds==y.flatten())*100:.2f}%")
                # Loss plot
                fig_loss, ax = plt.subplots()
                ax.plot(losses, color='pink')
                ax.set_title("Training Error per Epoch")
                st.pyplot(fig_loss)
                st.pyplot(plot_boundary(predict_func, X, y))

            else:
                # XOR NN training
                np.random.seed(42)
                W1 = np.random.randn(2,2); b1=np.zeros((1,2))
                W2 = np.random.randn(2,1); b2=np.zeros((1,1))
                losses = []
                for _ in range(epochs):
                    hidden = sigmoid(np.dot(X,W1)+b1)
                    out = sigmoid(np.dot(hidden,W2)+b2)
                    loss = np.mean((y-out)**2)
                    losses.append(loss)
                    d_out = (out-y)*out*(1-out)
                    d_W2 = np.dot(hidden.T,d_out); d_b2=np.sum(d_out,axis=0,keepdims=True)
                    d_hidden = np.dot(d_out,W2.T)*hidden*(1-hidden)
                    d_W1 = np.dot(X.T,d_hidden); d_b1=np.sum(d_hidden,axis=0,keepdims=True)
                    W2 -= lr*d_W2; b2-=lr*d_b2; W1-=lr*d_W1; b1-=lr*d_b1
                predict_func = lambda inp: np.round(forward_xor(inp,W1,b1,W2,b2))
                preds = predict_func(X)
                st.subheader("XOR Predictions (Trained)")
                st.write("Predictions:", preds.flatten().tolist())
                st.write(f"Accuracy: {np.mean(preds==y.flatten())*100:.2f}%")
                # Loss plot
                fig_loss, ax = plt.subplots()
                ax.plot(losses, color='pink')
                ax.set_title("MSE Loss Curve")
                st.pyplot(fig_loss)
                st.pyplot(plot_boundary(predict_func, X, y))
