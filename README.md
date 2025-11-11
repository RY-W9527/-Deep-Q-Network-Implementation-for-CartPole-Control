# Value Function Approximation — Linear Q-Learning & Deep Q-Network (DQN)

This repository contains the implementation for **Value Function Approximation** from the Reinforcement Learning course.  
The tutorial explores **linear function approximation** and **Deep Q-Networks (DQN)** on the **CartPole-v1** environment using **Gymnasium** and **PyTorch**.

---

## Learning Objectives

By completing this tutorial, you will learn to:

- Apply **stochastic gradient descent (SGD)** to value function learning.  
- Implement **semi-gradient TD learning** with both linear and neural function approximators.  
- Understand and implement **experience replay** and **target networks** for stabilizing training.  
- Train and evaluate a **DQN agent** capable of solving the **CartPole** environment.

---

## Project Structure

```
.
├── CartPole_DQN.py         # Main implementation (Linear Q-Learning + DQN)
├── 70028_4_spec.pdf             # Tutorial specification and assignment guide
├── linear_q-learning_results.png# Training results for Linear Q-Learning
├── deep_q-network_(dqn)_results.png # Training results for DQN
└── README.md                    # Project documentation (this file)
```

---

## Implementation Overview

### **1️ Linear Q-Learning (Value Function Approximation)**

- Implements **semi-gradient TD learning** using hand-crafted **polynomial features**.
- Uses a **linear Q-function**:  
  \( Q(s, a) = w_a^T \phi(s) \)
- Employs **ε-greedy** exploration for balancing exploration and exploitation.
- Trains until the agent consistently achieves the **solved threshold** (average reward ≥ 195).

### **2️ Deep Q-Network (DQN)**

- Extends Q-learning with a **neural network** to approximate \( Q(s,a) \).
- Introduces key DQN components:
  - **Experience Replay Buffer**: decorrelates samples.
  - **Target Network**: stabilizes learning.
  - **Mini-batch Updates**: improves sample efficiency.

---

## Dependencies

Make sure you have the following installed:

```bash
pip install gymnasium torch matplotlib numpy
```

---

## Running the Tutorial

Run both parts directly from the Python file:

```bash
python  CartPole_DQN.py
```

The script will:
1. Train a **Linear Q-Learning** agent.  
2. Train a **Deep Q-Network (DQN)** agent.  
3. Save plots for both training results automatically.

---


## Reflection

| Method | Advantages | Disadvantages |
| ------ | ---------- |-------------- |
| **Linear Q-Learning**    | - Simple and fast to compute<br>- Requires less data<br>- Easy to interpret and debug   | - Limited expressiveness<br>- May underfit complex problems<br>- Slower convergence on nonlinear tasks   |
| **Deep Q-Network (DQN)** | - Can approximate complex nonlinear Q-functions<br>- Learns more general and powerful representations<br>- Experience replay improves data efficiency | - Computationally expensive<br>- Requires more hyperparameter tuning<br>- Less interpretable; risk of instability |

---

## References

- [Gymnasium: CartPole Environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)  
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)  
- Krishnan, S. (2021). *Reinforcement Learning Tutorial Series*.

---

