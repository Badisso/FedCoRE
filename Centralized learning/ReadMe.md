# Centralized FedCoRE Training on WISDM Dataset

## 📜 Overview
This project provides the code to train and evaluate a centralized version of the **FedCoRE** framework for Human Activity Recognition (HAR). The model is trained on a pre-processed subset of the public WISDM dataset, which is automatically segmented, normalized, and split using a 5-fold cross-validation strategy.

---

## 🚀 Getting Started

### Prerequisites
Before running the code, ensure you have installed the required libraries using CI-install.sh:

### Setup and Data Preparation

1.  **Download the Dataset**: First, download the raw WISDM dataset and place the file `WISDM_ar_v1.1_raw.txt` in the project's root directory (if not exists already).

2.  **Generate Data Folds**: Run the `generate_json_data.py` script. This script will load the raw text file, process it into fixed-size windows, perform a 5-fold stratified split, and save the data into the required JSON format.

    ```bash
    python generate_json_data.py
    ```

    This command creates the `data/train` and `data/test` directories, populating them with the JSON files needed for training (e.g., `fold_1_train.json`, `fold_1_test.json`, etc.).

-----

## ⚙️ How to Run the Training

The main script to start the training process is `main.py`.

To properly evaluate the model using 5-fold cross-validation, you must run the training script five times, once for each fold. You can do this by modifying the `fold_idx` value inside the `load_data` function in `main.py` before each run.

### Running a Single Fold

1.  Open `main.py`.
2.  Locate the `load_data` function.
3.  Change the `fold_idx` argument to the desired fold number (e.g., `fold_idx=1` for the first fold, `fold_idx=2` for the second, and so on up to 5).
    ```python
    # Inside main.py in the load_data function
    ...
    class_num = load_partition_data_fed_wisdm2011(batch_size=args.batch_size, fold_idx=1) # Change 1 to 2, 3, 4, 5 for each run
    ...
    ```
4.  Run the script from your terminal:
    ```bash
    python main.py
    ```

Repeat this process for all 5 folds to get a complete cross-validation result.

**Note**: The best model from each run is saved as `best_model.pth`. This file will be **overwritten** by the next run. If you want to keep the best model from each fold, you should modify the `centralized_trainer.py` script to save the model with a fold-specific name.

-----

## 📂 Codebase Structure

  * `main.py`: The main executable script. It handles argument parsing, initializes the dataset, model, and trainer, and launches the training process.
  * `generate_json_data.py`: This is the data preparation script. It loads the raw `WISDM_ar_v1.1_raw.txt`, cleans it, creates 30-step windows, and saves the data into 5 stratified folds.
  * `data_loader.py`: Contains the `load_partition_data_fed_wisdm2011` function, which loads the pre-processed JSON files for a specific training fold.
  * `centralized_trainer.py`: Defines the `CentralizedTrainer` class, which manages the complete training and evaluation loop, including optimization, loss calculation, and saving the best model.

<!-- end list -->

```
```