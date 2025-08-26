
# 🧠 Neuron_Gates  

🔗 Live Demo: [https://neurongates.streamlit.app/](https://neurongates.streamlit.app/)  

---

## 📜 Overview  
This project was created for the **"Build a Single Neuron from Scratch to Train Logic Gates"** challenge.  
It focuses on building a **Perceptron** — the simplest type of artificial neuron — completely from scratch using **Python and NumPy**.  

The goal of the project is to show how a perceptron can learn to mimic **logic gates** such as **AND, OR, NAND, and XOR**.  
It also highlights the perceptron’s limitation when trying to solve the **XOR gate problem**, which cannot be learned by a single neuron.  

With this project, you can:  
- Train the perceptron on different logic gates  
- Watch how the learning process updates the decision boundary  
- Test the trained model with your own inputs  

In short, this project acts as a **learning tool** to understand:  
- How a perceptron works  
- Why some problems are solvable with a single neuron (**linear separability**)  
- Why more advanced networks (**multi-layer perceptrons**) are needed for harder problems like XOR  

---

## 🔍 Observations  
- The perceptron **successfully learns** linearly separable gates (AND, OR, NAND).  
- It **fails to classify XOR**, demonstrating the limitation of single-layer perceptrons.  
- Training progress is visible via **decision boundary shifts** across epochs.  
- Hyperparameters like **learning rate and epochs** significantly impact convergence.  
- Shows why **multi-layer networks (MLPs)** are required to solve XOR.  

---

## 📦 Deliveries  
✅ **Python Implementation** – A perceptron built completely from scratch using NumPy.  
✅ **Logic Gate Training** – Train the neuron on AND, OR, NAND, and XOR gates.  
✅ **Visualization** – Real-time plots of weight updates and decision boundaries.  
✅ **Interactive App** – Run and test via Streamlit and Pygame interfaces.  
✅ **Documentation** – Includes project report with terminology, problem statement, and learning rules.  
✅ **Educational Value** – Demonstrates linear separability and XOR limitations.  

---

## 🚀 How to Run Locally  

### 1. Prerequisites  
- Python 3.7+  
- Git  

### 2. Clone Repository  
```bash
git clone https://github.com/Swasti2004/Neuron_from_scratch.git
cd Neuron_Gates
````

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python gates-app.py
```

Or explore Jupyter Notebooks:

* `xor.ipynb` → XOR demonstration
* `train1.ipynb` → Training walkthrough

---


## 📁 File Structure  
```

Neuron\_Gates/
├── Documentation.docx     # Detailed write-up
├── LICENSE                # MIT License file
├── README.md              # Project description
├── requirements.txt       # Dependencies (NumPy, Streamlit, Pygame, etc.)
├── gates-app.py           # Main application
├── interaction.py         # User interaction handling
├── xor.ipynb              # XOR demo in notebook form
├── train1.ipynb           # Training steps and visuals
└── .devcontainer/         # Dev container setup

```


## 📊 Example Visuals

* Weight updates across epochs
* Decision boundary evolution
* XOR failure demonstration

---

## 📚 Learning Outcomes

Through this project, you will:

* Understand how a **Perceptron** works
* Explore **linear separability** in logic gates
* See why XOR requires **multi-layer perceptrons**
* Experiment with **interactive training**

---

## 🛠️ Tech Stack

* **Python** – Core implementation
* **NumPy** – Matrix operations & perceptron logic
* **Matplotlib** – Decision boundary visualization
* **Streamlit** – Interactive UI
* **Pygame** – Visual interaction & animations

---



## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repo
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Push & create a PR

---

## 👩‍💻 Authors

### Disha Shrivastava

🔗 [GitHub](https://github.com/Disha-shrivastava26) • [LinkedIn](www.linkedin.com/in/disha-shrivastava-21a697274)

### Swasti Jain

🔗 [GitHub](https://github.com/Swasti2004) • [LinkedIn](https://www.linkedin.com/in/swasti-jain2004/)

---

## 📜 License

Licensed under the **MIT License** – free to use and modify.

---


```
