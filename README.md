
# Crime Classification and Punishment Recommendation using BERT

## Introduction:
This project aims to automate the process of classifying crime news articles and generating suitable punishments based on their severity. The workflow involves utilizing a BERT (Bidirectional Encoder Representations from Transformers) model for crime classification and then recommending punishments using a dataset of predefined punishments. The classification model is trained on crime news data labeled with specific crime categories. Upon classification, the project suggests punishments by matching the classified crime with similar instances in a punishment dataset.

## Dataset:
The dataset used in this project consists of crime news articles paired with their corresponding crime labels. These labels include categories such as Murder, Kidnap, Drug, Fraud, etc. Additionally, a separate dataset containing predefined punishments is utilized for punishment recommendation.

<img width="322" alt="Dataset" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/6a3e3dbf-ff84-4ffa-93e7-3fb009175b00">

### Punishment Dataset

<img width="876" alt="judge" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/3af28949-3c22-4071-9907-6ffe4e60f2c4">


## Preprocessing:(Optional in our case)
Preprocessing plays a crucial role in preparing the data for model training. In this project, several preprocessing techniques are applied to the crime news data:

- **Tokenization:** Tokenization involves breaking down the raw text into individual tokens or words. In our case, we tokenize the crime news articles to convert them into a format suitable for input into the BERT model. Tokenization allows the model to understand the textual data at the word level, capturing semantic meaning and context.

  <img width="461" alt="Token" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/5a69978f-1dc4-4fac-9254-a883246fd64d">


- **Padding:** To ensure uniformity in input size, we apply padding to the tokenized sequences. Padding involves adding special tokens to the sequences so that they all have the same length. This is necessary because BERT, like many deep learning models, requires inputs of fixed dimensions. The padding ensures that shorter sequences match the length of the longest sequence in the dataset, enabling efficient batch processing during training.

- **Encoding:** Encoded training and validation texts using a tokenizer with truncation and padding, returning PyTorch tensors. train_texts and val_texts are input texts for training and validation, respectively."

  <img width="585" alt="encode" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/e31615f3-0289-4928-a70d-9cc549e21dc0">


- *As we use the Bert Model, it also provides a Bert tokenizer, which almost did all preprocessing work for us*

## Model Architecture:
The core of this project's model architecture is based on BERT, a state-of-the-art transformer-based model. BERT is pre-trained on a large corpus of text data and fine-tuned for sequence classification tasks. The BERT model used here is a pre-trained version of BERT-base-uncased.

<img width="668" alt="model" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/8f0b4591-001b-4bf6-9839-a1918448292b">


## Feature Engineering:
Feature engineering involves creating input features that best represent the underlying patterns in the data. In our project, feature engineering mainly revolves around preparing the input data for the BERT model:

- **Tokenization:** As mentioned earlier, tokenization is a crucial step in feature engineering. It converts the raw text data into a format understandable by the model, breaking it down into tokens or words.

- **Input IDs and Attention Masks:** After tokenization, we convert the tokenized sequences into input IDs, which are numerical representations of the tokens. Additionally, we generate attention masks to indicate which tokens the model should pay attention to and which ones are padding tokens. This helps the model focus on relevant information while ignoring padded tokens during training.

  

## Hyperparameter Tuning:
Hyperparameters are parameters that define the structure and behavior of the model during training. Tuning these hyperparameters is essential for optimizing model performance. In our project, we focus on the following hyperparameters:

- **Learning Rate:** The learning rate controls the step size during gradient descent optimization. We set the learning rate to 5e-5, a commonly used value for fine-tuning BERT models. A higher learning rate may lead to faster convergence but risks overshooting the optimal solution, while a lower learning rate may slow down training.

- **Batch Size:** It is 8 is our case. The batch size determines the number of samples processed in each iteration of training. We experiment with different batch sizes to find the optimal balance between computational efficiency and model performance. Larger batch sizes can accelerate training but may require more memory, while smaller batch sizes may lead to more stable convergence.

- **Number of Epochs:** 3 in our case .Epochs refer to the number of times the entire dataset is passed through the model during training. We tune the number of epochs to ensure sufficient training without overfitting or underfitting the data. Early stopping techniques may also be employed to prevent overfitting if necessary.

  <img width="281" alt="epouch" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/0de4a43d-5882-4699-90a2-0b3cf0e253e2">


## Deep Learning Models:
The deep learning model employed in this project is BERT (Bidirectional Encoder Representations from Transformers) for sequence classification. BERT is renowned for its ability to capture contextual information and is well-suited for natural language processing tasks.

## Performance Measures:
The performance of the classification model is evaluated using accuracy as the primary metric. The accuracy achieved on the validation set is reported to be 99%, indicating the effectiveness of the model in classifying crime news articles.

<img width="180" alt="acc" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/9ddd1733-271d-454f-9e00-638adad67d41">


## Model Evaluation:
The trained model is evaluated on unseen crime news articles to classify their crime category. Additionally, punishments are recommended based on the severity of the classified crime. The recommended punishments are obtained by matching the classified crime with similar instances in a punishment dataset.

<img width="325" alt="measure" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/9c5cce23-8704-49ef-b65d-0aa556540308">

## Prediction:
To evaluate our model we provide it with a prompt so it predicts the crime.

<img width="317" alt="pre" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/fbe4a87b-c59a-4763-90f0-d3acc26dfbc2">

## Punishment:
At the end after predicting the crime label we use this label and prompt to announce the punishment of that crime using similarity measures like fuzzywuzzy and Cosine similarity.
Showing top 5 Judgments that can possibly be implemented in that situation

<img width="466" alt="punis" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/031ed629-006d-4479-88b6-85723ccd0ae3">

## Trained Model
The training of the model takes up to 1.5 hours in the free version of Google Collab, So to make sure we save time we saved the trained model in the "pth" file so we do not have to train it again and again instead just load it after download "pth file"

<img width="619" alt="model load" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/d872ba68-e96f-4b6d-b87c-2d0a67ce64cf">

here "bert_sentiment_model.pth" is our own trained classifier. Its link will be shared in file.


## Interface

We use streamlit to create an interface for our model.

<img width="921" alt="int 1" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/9b671ba0-0965-411f-9559-b2844ef6eddc">


<img width="926" alt="int 2" src="https://github.com/mubashirX/Crime-Detection/assets/112867358/db8943c0-1b35-42c9-a7a1-8107eeb67876">




# Conclusion:
In conclusion, this project demonstrates the effectiveness of utilizing BERT for crime classification and punishment recommendation. By automating these tasks, law enforcement agencies and legal practitioners can streamline the process of identifying crimes and recommending suitable punishments, thereby enhancing efficiency and accuracy in the criminal justice system.

---
