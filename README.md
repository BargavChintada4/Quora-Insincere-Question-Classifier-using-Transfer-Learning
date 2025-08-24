# Quora Insincere Question Classifier using Transfer Learning

This project demonstrates an efficient approach to classifying Quora questions as sincere or insincere using various pre-trained text embedding models from TensorFlow Hub. The primary technique employed is transfer learning, where we leverage powerful, pre-trained models to generate numerical representations (embeddings) of text data, which are then used to train a simple downstream classifier.

### Project Overview

The goal is to build a binary classifier that can accurately identify insincere questions on the Quora platform. Insincere questions are defined as those intended to make a statement rather than seeking genuine answers.

This implementation focuses on comparing the performance of five different text embedding models by:
1.  **Pre-computing embeddings** for the entire dataset for each model. This is a highly efficient workflow as the computationally expensive feature extraction step is performed only once.
2.  Training a separate, simple neural network classifier on these pre-computed numerical embeddings.

### Models Compared

The following pre-trained text embedding models from TensorFlow Hub were used and evaluated:
* **GNews Swivel (20-dim)**: `https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1`
* **NNLM (50-dim)**: `https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1`
* **NNLM (128-dim)**: `https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1`
* **Universal Sentence Encoder (USE)**: `https://tfhub.dev/google/universal-sentence-encoder/4`
* **Universal Sentence Encoder - Large**: `https://tfhub.dev/google/universal-sentence-encoder-large/5`

### Methodology

1.  **Dataset**: The project uses the [Quora Insincere Questions Classification dataset](https://www.kaggle.com/c/quora-insincere-questions-classification). The data is loaded directly from an online source.
2.  **Preprocessing**: A small subset of the data (1% for training, 0.1% for validation) is used for demonstration and quick experimentation.
3.  **Embedding Generation**: For each of the five models, a `hub.KerasLayer` is used to convert the text questions from the training and validation sets into fixed-size numerical vectors. The pre-trained layers are kept frozen (`trainable=False`).
4.  **Classifier Architecture**: A simple feed-forward neural network is built to perform the classification on the generated embeddings. The architecture consists of:
    * An Input Layer with a shape corresponding to the embedding dimension (e.g., 512 for USE).
    * A Dense hidden layer with 256 neurons (`ReLU` activation).
    * A Dense hidden layer with 64 neurons (`ReLU` activation).
    * A Sigmoid output layer for binary classification.
5.  **Training**: Each classifier is trained on its corresponding embeddings using the `Adam` optimizer, `BinaryCrossentropy` loss, and `BinaryAccuracy` metric. Early stopping is employed to prevent overfitting.

### How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Install dependencies:** Make sure you have TensorFlow, TensorFlow Hub, Pandas, and Scikit-learn installed.
    ```bash
    pip install tensorflow tensorflow-hub pandas scikit-learn tensorflow-docs
    ```
3.  **Run the Jupyter Notebook:** Launch and run the cells in `Transfer_Learning_NLP_97_.ipynb`.

### Results

The performance of each model was tracked and plotted.

* **Accuracy**: The models based on the **Universal Sentence Encoder (USE)** achieved the highest validation accuracy (~95%), converging in fewer epochs than the other models.
* **Loss**: The USE models also demonstrated the lowest validation loss, indicating better generalization on unseen data.

This experiment shows the power of using high-quality pre-trained embeddings for NLP tasks, allowing for the creation of high-performance models with minimal training time and computational resources.
