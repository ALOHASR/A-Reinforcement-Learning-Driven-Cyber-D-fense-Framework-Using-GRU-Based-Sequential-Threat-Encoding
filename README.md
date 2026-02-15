🛡️ Reinforcement-Learning-Driven Cyber-Defense Framework
Using GRU-Based Sequential Threat Encoding
📌 Overview

This project proposes an intelligent cyber-defense framework that combines GRU-based sequential threat encoding with Reinforcement Learning (RL) for adaptive intrusion detection and response.

The system learns to:

Encode sequential network traffic behavior using GRU

Model evolving attack patterns

Make optimal defense decisions using reinforcement learning

The model is trained and evaluated using the CIC_IDS_2017_CSV dataset.

🎯 Problem Statement

Traditional intrusion detection systems are:

Static and signature-based

Unable to adapt to evolving threats

Inefficient in handling sequential attack behavior

This project introduces a learning-based adaptive defense mechanism that dynamically selects optimal defensive actions based on observed network traffic patterns.

🧠 System Architecture

Data Preprocessing

Dataset: CIC_IDS_2017_CSV

Feature cleaning

Normalization

Handling missing and infinite values

Label encoding

Sequential Threat Encoding (GRU)

Input: Network flow feature sequences

Model: GRU-based neural network

Output: Encoded threat state representation

Reinforcement Learning Agent

State: GRU-encoded threat vector

Actions:

Allow traffic

Block traffic

Throttle

Isolate source

Reward Function:

+Reward for correct mitigation

−Penalty for false positives/negatives

Adaptive Defense Decision Engine

Learns optimal defense policies

Improves over time through interaction

📊 Dataset

Dataset Used: CIC_IDS_2017_CSV

Contains realistic network traffic

Includes attack types:

DoS

DDoS

PortScan

Brute Force

Web Attacks

Extracted flow-based features

Multi-class classification setup

⚙️ Tech Stack

Python

PyTorch / TensorFlow

NumPy

Pandas

Scikit-learn

Matplotlib

🏗️ Project Structure
├── data/
├── preprocessing/
├── models/
│   ├── gru_encoder.py
│   ├── rl_agent.py
├── training/
├── evaluation/
├── utils/
├── README.md

🚀 Installation
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

▶️ Usage
Train GRU Encoder
python train_gru.py

Train RL Agent
python train_rl.py

Evaluate Model
python evaluate.py

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Reward Convergence

False Positive Rate

🔬 Research Contribution

✔️ Sequential modeling of cyber threats using GRU
✔️ Integration of deep learning with reinforcement learning
✔️ Adaptive defense policy learning
✔️ Reduced false positive rate compared to static IDS

📚 Future Work

Integration with real-time SDN environments

Deployment in edge computing environments

Multi-agent reinforcement learning

Online continual learning for zero-day attacks

👤 Author

Sree
Research Focus: AI-driven Cyber Defense Systems
