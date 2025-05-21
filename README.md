# NLP Sentiment Classification Using BiLSTM and Activation Function Tuning

**Author: May Cooper**

## Overview

This NLP classification project applies a BiLSTM-based neural network to classify sentiment in short user-generated reviews. It uses pre-trained GloVe word embeddings for semantic representation and compares the effects of three activation functions—ReLU, sigmoid, and tanh—on model performance. The goal is to evaluate how activation functions influence classification accuracy and generalization on real-world text data.

---

## Tools and Technologies

* **Python**
* **TensorFlow / Keras**
* **NumPy / Pandas**
* **Matplotlib / Seaborn**
* **GloVe Pre-trained Word Embeddings**

---

## What is BiLSTM?

BiLSTM (Bidirectional Long Short-Term Memory) is a type of recurrent neural network that processes input sequences both forward and backward. This structure helps the model capture context from both directions—particularly useful for understanding sentiment in natural language.

In this project, a BiLSTM is used with an embedding layer initialized from GloVe vectors, allowing the model to incorporate pre-learned word meanings and relationships.

---

## Project Objectives

* **Primary Goal**: Classify the sentiment of short user reviews using a neural network.
* **Research Question**: What is the effect of different activation functions (ReLU, sigmoid, tanh) on the performance of a BiLSTM-based sentiment classification model?
* **Workflow:**

  1. Clean and preprocess text data from multiple labeled review datasets.
  2. Tokenize and pad sequences to a fixed maximum length.
  3. Load GloVe vectors into an embedding layer.
  4. Build and train a BiLSTM model with different activation functions.
  5. Compare performance across activations using validation accuracy and loss.

---

## Dataset Summary

This project uses three datasets containing labeled user reviews:

* **Amazon product reviews**
* **IMDb movie reviews**
* **Yelp restaurant reviews**

Each dataset contains 1,000 sentences labeled with:

* **0** = negative sentiment
* **1** = positive sentiment

All datasets are combined, cleaned, and tokenized for training.

### Sample of Preprocessed Data

| ReviewText                                               | Sentiment |
| -------------------------------------------------------- | --------- |
| Good case, excellent value.                              | 1         |
| Very, very slow-moving and aimless.                      | 0         |
| I tried the Cape Cod ravioli with cranberry...mmm!       | 1         |
| The movie totally grates on my nerves.                   | 0         |
| The sweet potato fries were very good and seasoned well. | 1         |

---
### Exploratory Data Analysis

#### Sentiment Label Distribution
![image](https://github.com/user-attachments/assets/0b6dc36c-e36f-4ef1-9999-9da1d07ecde4)

The dataset is relatively balanced, containing a similar number of positive (~390) and negative (~360) reviews. This ensures the model is not biased toward one class during training.

#### Word Cloud of Most Frequent Words
![image](https://github.com/user-attachments/assets/12c73c74-05d9-4e8f-a452-93b815c1de1a)

Common words such as **“movie”**, **“good”**, **“bad”**, **“plot”**, and **“character”** dominate the vocabulary. These terms reflect the emotionally charged language typical of user reviews and are valuable for sentiment classification.

#### Sentence Length Distribution

![image](https://github.com/user-attachments/assets/8a393102-0ded-4c03-aee1-912770b5c343)

The distribution of sentence lengths shows that most reviews are short, with the 95th percentile at **34 tokens**. This value was selected as the **maximum sequence length** for padding to ensure coverage while maintaining training efficiency.

---
## Model Architecture

1. **Embedding Layer** – 100-dimensional GloVe vectors
2. **BiLSTM Layer** – 128 hidden units
3. **Dense Layer** – with variable activation (ReLU, sigmoid, tanh)
4. **Dropout Layer** – for regularization
5. **Output Layer** – 1 unit with sigmoid activation for binary classification

---

## Performance Comparison

![image](https://github.com/user-attachments/assets/80408962-c622-4fd7-89a0-36d3a40c499e)

**Best Model (ReLU activation):**

* **Train Accuracy**: 86.6%
* **Validation Accuracy**: 72.5%
* **Train Loss**: 0.4697
* **Validation Loss**: 0.6986

**Other Activations:**

* **Sigmoid**: 89.3% train, 70.0% validation
* **Tanh**: 86.3% train, 71.7% validation

All models were trained for up to 30 epochs with early stopping.

---

## Visual Analysis and Interpretations

### Training Progress and Early Convergence
![image](https://github.com/user-attachments/assets/d431c213-16da-4094-82f3-054ad7ad63ae)

The training log shows the BiLSTM model trained with ReLU activation over 14 epochs before early stopping was triggered. The model consistently improved across epochs:

- **ReLU** achieved **validation accuracy up to 75.8%** by Epochs 7, 10, and 11, with **training accuracy plateauing around 88–89%** and **validation loss decreasing steadily**.
- **Sigmoid** showed **higher training accuracy (up to 89%)** but experienced **more fluctuation in validation accuracy** and **slightly worse generalization**, indicated by rising validation loss after epoch 13.
- **Tanh** reached the **highest training accuracy (90.3%)** with a **validation accuracy plateau around 71.7–72.5%**, and relatively stable validation loss.
  
### Validation Accuracy and Loss by Activation Function
![image](https://github.com/user-attachments/assets/06de689c-2a93-4b67-bb6f-676d3ecb1461)

The comparison charts below show how different activation functions affect the model’s performance on the validation set:

- **ReLU** achieved the highest validation accuracy (~72.5%) with moderate validation loss.
- **Sigmoid** had the lowest validation loss but slightly lower accuracy (~70.0%).
- **Tanh** performed comparably to ReLU in accuracy (~71.7%) but had the highest validation loss.

These results suggest that **ReLU** offers the best trade-off between accuracy and loss for this binary sentiment classification task, while **sigmoid** may provide slightly better generalization depending on the evaluation criteria.

### Training vs. Validation Accuracy and Loss
![image](https://github.com/user-attachments/assets/16e9b037-2a8c-453c-bae9-3d5c01bc3f04)
These plots highlight how the model performed during training across different activation functions:

- **ReLU** and **tanh** showed more consistent validation accuracy compared to **sigmoid**, which had the largest gap between training and validation performance.
- **Sigmoid** had the **lowest training loss**, but also the **highest overfitting gap**, suggesting the model may have memorized the training data more than generalizing.
- **Tanh** showed the **lowest training loss overall**, but validation performance was slightly less stable.
- **ReLU** delivered a strong balance of generalization and performance, making it a preferred activation for this task.

Overall, the results support **ReLU** as the most balanced option, with **tanh** as a close second depending on deployment goals (e.g., training stability vs. loss minimization).

---

## Conclusion

This project demonstrates that BiLSTM-based sentiment classification benefits from careful activation function tuning. While all three tested functions achieved reasonable performance, ReLU yielded the most stable validation accuracy. The pipeline is robust for binary sentiment tasks on short text, with further potential for improvement via attention mechanisms or transformer-based upgrades.

---

## Future Work

* Extend to multiclass sentiment (e.g., neutral, mixed)
* Use GRU or Transformer-based alternatives
* Add attention layers for interpretability
* Deploy using Flask/FastAPI with Docker for real-time sentiment services

---

## Deployment Notes

* Model saved as: `sentiment_model_relu.keras`
* Can be loaded and served using TensorFlow SavedModel or `keras.models.load_model()`
* Optional: wrap in API for real-time inference
