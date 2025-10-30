# 🏦 Loan Approval Optimization with Deep Learning and Offline Reinforcement Learning

## 📘 Overview
This project explores two complementary approaches to **credit risk modeling and loan approval decision-making** using data from **Lending Club**.  

1. **Model 1: Predictive Deep Learning Classifier**  
   - Predicts the probability that a loan applicant will **default**.  
   - Uses a multi-layer neural network trained with binary cross-entropy.  

2. **Model 2: Offline Reinforcement Learning Agent**  
   - Learns an **approval policy** that maximizes long-term expected financial returns.  
   - Trained on historical data using **Conservative Q-Learning (CQL)** from `d3rlpy`.  

This project demonstrates how traditional **supervised learning** and **reinforcement learning** can complement each other in **credit risk management**.

---

## 📊 Dataset
The dataset is derived from **LendingClub’s public loan data (2007–2018)**.  
Each record represents a loan application with detailed applicant and loan attributes.

| Component | Description |
|------------|-------------|
| **Features (X)** | Borrower’s financial and credit profile (income, FICO score, DTI, loan amount, etc.) |
| **Target (y)** | `loan_status`: {0 = Fully Paid, 1 = Defaulted} |
| **Size** | ~2.2M records (subset used for training and RL simulation) |

---

## 🧬 Model 1: Deep Learning Classifier

### 🎯 Goal
Predict the **probability of loan default** and evaluate performance using AUC and F1-score.

### 🧩 Model Architecture
```python
Input → Dense(512, ReLU) → BN → Dropout(0.4)
      → Dense(256, ReLU) → BN → Dropout(0.4)
      → Dense(128, ReLU) → LayerNorm → Dropout(0.3)
      → Dense(64, ReLU)
      → Dense(1, Sigmoid)
```

### ⚙️ Training Setup
- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam (lr=1e-3 with ReduceLROnPlateau)  
- **Regularization:** Dropout + L2 + BatchNorm  
- **EarlyStopping:** Based on validation AUC  

### 📈 Results
| Metric | Value |
|---------|-------|
| Test AUC | ~0.72 |
| Test F1-Score | ~0.43 |

### ✅ Interpretation
- The model performs **strongly on discrimination (AUC)** but faces class imbalance.  
- AUC and F1-score capture the model’s ability to balance **precision vs recall** and **ranking quality** — crucial for credit risk tasks.

---

## 🤖 Model 2: Offline Reinforcement Learning Agent

### 🎯 Goal
Learn a policy that **decides whether to approve or deny** a loan to **maximize long-term profit**.

### 🥱 RL Environment
| Element | Definition |
|----------|-------------|
| **State (s)** | Loan applicant features |
| **Action (a)** | {0: Deny Loan, 1: Approve Loan} |
| **Reward (r)** | `+loan_amnt * int_rate` if approved & fully paid, `-loan_amnt` if defaulted, else `0` |

### ⚙️ Implementation
- Framework: [`d3rlpy`](https://github.com/takuseno/d3rlpy)
- Algorithm: **Conservative Q-Learning (CQL)** — robust for offline datasets  
- Dataset: Constructed as an `MDPDataset` from preprocessed loan data  

### 📈 Results
| Policy | Avg Reward per Loan | Total Return |
|---------|--------------------|---------------|
| **Learned RL Policy** | ~Higher than baseline | — |
| **Always Approve** | High variance, lower mean |
| **Always Deny** | Zero reward (baseline) |

### 💡 Interpretation
- The RL agent learns to **balance risk and reward**, sometimes approving loans the DL model would deny if expected profit outweighs default risk.  
- The key metric is **Estimated Policy Value (EPV)** — the average reward under the learned policy.  
  It reflects **expected business profit**, not just accuracy.

---

## ⚖️ Comparison: DL vs RL

| Aspect | Deep Learning Model | RL Agent |
|--------|--------------------|----------|
| **Learning Type** | Supervised | Offline Reinforcement |
| **Output** | Default probability | Loan approval decision |
| **Objective** | Maximize prediction accuracy | Maximize long-term financial reward |
| **Metric** | AUC, F1-score | Estimated Policy Value (EPV) |
| **Interpretation** | Risk estimation | Policy optimization |

### Example Decision Difference
- A borderline applicant:  
  - **DL Model:** Flags as high-risk (probability > 0.5 → reject)  
  - **RL Agent:** Approves due to potential profit (`reward = loan_amnt * int_rate`)  
- The RL agent is **profit-driven**, while the DL model is **risk-averse**.

---

## 🚀 Future Directions

1. **Deploying Hybrid Policy**
   - Combine DL’s risk probability with RL’s reward-driven policy for **explainable decision-making**.

2. **Advanced Offline RL**
   - Experiment with algorithms like **TD3+BC**, **IQL**, or **BCQ** to improve stability.

3. **Data Enrichment**
   - Include **macro-economic variables**, **transaction history**, or **real-time payment data**.

4. **Ethical & Regulatory Analysis**
   - Ensure fairness, transparency, and compliance in automated loan decision systems.

---

## 🧩 Project Structure
```
├── loan_data_processed.csv       # Preprocessed dataset
├── dl_model.ipynb          # Deep Learning model training
├── rl_model.ipynb          # Offline RL agent training
├── analysis_comparison.ipynb     # Comparison & visualization
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Setup & Execution

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/loan-approval-rl.git
cd loan-approval-rl

# 2️⃣ Install dependencies
pip install -r requirements.txt
```
Now run each Cell in the python notebooks

---

## 📚 References
- LendingClub Public Loan Dataset  
- Takuma Seno, *d3rlpy: Deep Offline Reinforcement Learning Library*, 2022  
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd Ed.

---

## 🏁 Author
**Nikhil Goyal**    
📧 [goyalnikhil883@gmail.com]  