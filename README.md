# FedCoRE – Effective Federated Learning for Constrained RESTful Environments in the AIoT

## 📜 Overview

**FedCoRE** is a practical Federated Learning (FL) framework designed for resource-constrained IoT devices in **Artificial Intelligence of Things (AIoT)** environments.
It leverages **standards for constrained RESTful environments**, such as the **Constrained Application Protocol (CoAP)**, to optimize communication, and uses **model quantization** to address computation and storage limitations.

FedCoRE has been implemented and tested on devices with **256 KB of RAM**, achieving significant reductions in communication costs while maintaining model accuracy.

This repository contains the **codebase and tool-chain** developed as part of the FedCoRE project, organized into:

* **Centralized Training** – Baseline training and evaluation of models in a non-federated setting.
* **Federated Training** – Implementation of FedCoRE and standard FL pipelines.
* **Dataset** – Data preparation scripts and raw datasets used in experiments.


## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Install required dependencies using the provided installation scripts in each sub-directory.
* Some experiments require **CoAP libraries** and **embedded device toolchains** (instructions will be added soon).

### Repository Structure

```
FedCoRE/
│
├── centralized_training/      # Centralized (non-FL) baseline experiments
│   └── README.md               # Instructions for centralized training
│
├── federated_training/         # FedCoRE and standard FL implementations
│   └── README.md               # Instructions for federated training
│
├── dataset/                    # Data used in the federated training
│   └── README.md               # description of the dataset
│
└── README.md                   # Main repository readme (this file)
```

Each sub-directory contains its **own README file** with detailed execution instructions.

---

## ⚙️ How to Run

### 1. Centralized Training

Navigate to the `centralized_training` directory and follow its README instructions to reproduce baseline results.

### 2. Federated Training (FedCoRE)

Navigate to the `federated_training` directory and follow its README to run FedCoRE on the target dataset.



---

## 📂 Codebase Notes

* The **centralized** and **federated** training modules are independent but share a similar data pipeline.
* Model configurations, hyperparameters, and training options are customizable in each module.
* The FedCoRE implementation supports **model quantization**, **communication compression**, and **CoAP-based messaging** for deployment on constrained IoT devices.

---

## 📌 License

This project is released for research purposes. Please cite when using the code.
