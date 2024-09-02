# LLM-Course-Projects

## CA1 Project Description

This project focuses on fine-tuning the BERT language model for various natural language processing tasks, such as text classification and question answering. The project is divided into two main parts:

### Word Embeddings and Masked Language Models (MLMs)

In this part, we utilize the `Gensim` library to work with `GloVe` word embeddings and visualize the results for each word. `GloVe` provides low-dimensional dense vectors representing word semantics, where the distance between word embeddings captures their semantic relationships. Following this, we explore Masked Language Modeling (MLM) with BERT, which captures contextual embeddings. MLM, used in models like BERT, involves training the model to predict masked words within sentences based on contextual cues. During training, tokens are randomly replaced with a special token `[MASK]`, and the model learns to predict the original tokens based on the surrounding context. This technique is foundational for downstream tasks after pre-training.

### Transfer Learning with BERT

In this part of the project, we use Hugging Face's `Transformers` library, which provides general-purpose architectures for natural language understanding. This allows us to leverage pre-trained models like `BERT` and perform experiments on top of them. After installing the necessary packages, we fine-tuned `BERT` for tasks such as text classification and question answering. Finally, we evaluated the performance of the fine-tuned models on both tasks.

