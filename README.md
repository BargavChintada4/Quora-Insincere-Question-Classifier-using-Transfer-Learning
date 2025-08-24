# Quora Insincere Question Classifier using Transfer Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A project to classify Quora questions as sincere or insincere using Transfer Learning by comparing various pre-trained text embedding models from TensorFlow Hub.

### Table of Contents
1.  [Project Objectives](#project-objectives)
2.  [Methodology](#methodology)
3.  [Results](#results)
4.  [How to Run the Project](#how-to-run-the-project)
5.  [Certification](#certification)
6.  [License](#license)

---

### Project Objectives

The primary goal of this project is to build an accurate and efficient binary classifier to identify insincere questions on the Quora platform. The project focuses on:
-   **Leveraging Transfer Learning:** Utilizing powerful, pre-trained NLP models to generate high-quality text embeddings.
-   **Model Comparison:** Systematically comparing the performance of five different text embedding models from TensorFlow Hub to determine the most effective one for this task.
-   **Efficient Workflow:** Implementing an optimized workflow by pre-computing embeddings to significantly speed up the training of the downstream classifier.

---

### Methodology

The process is broken down into two main stages: feature extraction and classification.

1.  **Dataset**: The project uses the [Quora Insincere Questions Classification dataset](https://www.kaggle.com/c/quora-insincere-questions-classification). For rapid prototyping, a stratified 1% sample is used for training and a 0.1% sample for validation.

2.  **Embedding Generation (Feature Extraction)**: The core of the project involves converting text questions into numerical vectors. Five pre-trained models from TensorFlow Hub were used for this, with their weights frozen (`trainable=False`):
    -   `gnews-swivel-20dim`
    -   `nnlm-en-dim50`
    -   `nnlm-en-dim128`
    -   `universal-sentence-encoder-4`
    -   `universal-sentence-encoder-large-5`

3.  **Classifier Architecture**: A simple but effective feed-forward neural network is trained on top of the pre-computed embeddings.
    -   **Input Layer**: Shape matches the embedding dimension of the chosen model.
    -   **Hidden Layers**: Two `Dense` layers with 256 and 64 neurons respectively (`ReLU` activation).
    -   **Output Layer**: A `Dense` layer with a single neuron and `Sigmoid` activation for binary classification.

The full implementation can be found in the Jupyter Notebook: [`Transfer_Learning_NLP_97_.ipynb`](./Transfer_Learning_NLP_97_.ipynb).

---

### Results

The performance of each model was tracked, and the results clearly indicate that the quality of embeddings is the most critical factor.

-   **Highest Accuracy**: The models based on the **Universal Sentence Encoder (USE)** achieved the highest validation accuracy, reaching approximately **95%**.
-   **Fastest Convergence**: The USE-based models also converged in far fewer epochs and achieved the lowest validation loss, demonstrating superior generalization.

For a detailed analysis and visualizations, please see the full [**Project Report**](./REPORT.md).

---

### How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow tensorflow-hub pandas scikit-learn tensorflow-docs
    ```
3.  **Run the Jupyter Notebook:**
    Launch Jupyter and run all cells in `Transfer_Learning_NLP_97_.ipynb`.

---

### Certification

This project was developed as part of my learning journey in Natural Language Processing. The skills applied here were solidified through the **[Name of Your Coursera Course/Specialization]** certification.

**[View My Certificate]**([YOUR_CERTIFICATE_LINK])

---

### License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
