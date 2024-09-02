# LLM-Course-Projects

## CA1 Project Description

This project focuses on fine-tuning the BERT language model for various natural language processing tasks, such as text classification and question answering. The project is divided into two main parts.

### Word Embeddings and Masked Language Models (MLMs)

In this part, we utilize the `Gensim` library to work with `GloVe` word embeddings and visualize the results for each word. `GloVe` provides low-dimensional dense vectors representing word semantics, where the distance between word embeddings captures their semantic relationships. Following this, we explore Masked Language Modeling (MLM) with BERT, which captures contextual embeddings. MLM, used in models like BERT, involves training the model to predict masked words within sentences based on contextual cues. During training, tokens are randomly replaced with a special token `[MASK]`, and the model learns to predict the original tokens based on the surrounding context. This technique is foundational for downstream tasks after pre-training.

### Transfer Learning with BERT

In this part of the project, we use Hugging Face's `Transformers` library, which provides general-purpose architectures for natural language understanding. This allows us to leverage pre-trained models like `BERT` and perform experiments on top of them. After installing the necessary packages, we fine-tuned `BERT` for tasks such as text classification and question answering. Finally, we evaluated the performance of the fine-tuned models on both tasks.

## CA2 Project Description

This computer assignment consists of two independent projects related to prompting techniques in large language models (LLMs). First, we work with the GPT-2 and explore different types of prompting. Then, we delve into the Soft Prompting method and evaluate its results.

### GPT-2 Prompting

In this project, we examine two methods of prompting for the GPT-2 model.

#### Single Sentence Prompting
After loading the model, we used a single sentence with ten tokens as the prompt for GPT-2 in the first method. In this experiment, we generated 190 new tokens, appending each new token to the previous sequence at every step. We measured the average token generation time, memory usage, and calculated the loss at each step of the experiment, then plotted all the collected data.

#### Batch Generation Prompting
In the second approach, we created four prompts with different lengths and applied a batch generation method. We repeated the above experimental steps, calculating the average token generation time, memory usage, and throughput per step to examine the performance of this method as well.

### Soft Prompting

Soft prompts are learnable tensors concatenated with the input embeddings, which can be optimized for a specific dataset. However, the downside is that they are not human-readable because these "virtual tokens" do not correspond to real words. In this project, we used `bert-fa-base-uncased` as our base model from Hugging Face, with the goal of utilizing 20 soft prompt tokens. After loading a dataset of 7,000 Persian sentences and performing preprocessing, we defined our Prompt Embedding Layer and completed the `initialize_embedding` and `forward` functions, replacing the model's embedding layer with our own layer. Following this, we implemented the `evaluation` function and `training loop` to examine the model's performance and results.

## CA3 


