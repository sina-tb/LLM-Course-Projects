# LLM-Course-Projects

## Table of Contents

1. [CA1 Project Description](#ca1-project-description)
   - [Word Embeddings and Masked Language Models (MLMs)](#word-embeddings-and-masked-language-models-mlms)
   - [Transfer Learning with BERT](#transfer-learning-with-bert)

2. [CA2 Project Description](#ca2-project-description)
   - [GPT-2 Prompting](#gpt-2-prompting)
   - [Soft Prompting](#soft-prompting)

3. [CA3 Project Description](#ca3-project-description)
   - [Chain-of-Thoughts](#chain-of-thoughts)
   - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
   - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)

4. [CA4 Project Description](#ca4-project-description)
   - [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
   - [Quantization and Instruct Tuning](#quantization-and-instruct-tuning)
   - [Evaluation](#evaluation)



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

## CA3 Project Description

This computer assignment consists of three independent projects related to large language models (LLMs). First, we explore the Chain-of-Thought concept in LLMs. Then, we delve into Parameter-Efficient Fine-Tuning (PEFT) methods and work with the `microsoft/phi-2` model for a question generation task. Finally, we explore the Retrieval-Augmented Generation (RAG) method and work with the LLaMa-2 model for an information retrieval task.

### Chain-of-Thoughts

In this project, after exploring the Tree-of-Thoughts and Self-Consistency concepts in LLMs, we use the `Phi-2` model to examine a question-answering task using Chain-of-Thought (CoT) reasoning and compare it to an approach without CoT. Then, we implement the Self-Consistency (SC) application in the CoT method and apply the CoT-SC method to the question-answering task.

### Parameter-Efficient Fine-Tuning (PEFT)

In this part of the assignment, we explore Parameter-Efficient Fine-Tuning (PEFT). First, we examine why PEFT is important for training LLMs and apply it to the `microsoft/phi-2` model for a question-answering task. We use the `Super-NaturalInstructions` dataset from the Hugging Face Hub and, after preprocessing, we select random samples from the test set. We then apply the `Alpaca template` to these samples and obtain model outputs. In the final phase of this project, we fine-tune the model using the Low-Rank Adaptation (LoRA) method.

### Retrieval-Augmented Generation (RAG)

In this part of the project, we focus on an application that fetches country-related information from a large language model. For this purpose, we use the `LLaMa-2-Chat-7B` model as our base. Initially, we create a simple chain that takes the name of a country as input and outputs its capital. We then enhance the chain to extract more detailed information by modifying the prompt to request data about a country's name, population, major cities, and capital in a structured `JSON` format. Next, we compare the performance of a TF-IDF Retriever and a Semantic Retriever system using the `Evaluate Retriever` function we implemented to measure the accuracy of the retrieved documents. Finally, after considering all the previous concepts and steps, we create a complete RAG chain and evaluate the results of this method in information retrieval tasks.

## CA4 Project Description

This computer assignment consists of three independent projects related to large language models (LLMs). First, we explore the concept of Reinforcement Learning from Human Feedback (RLHF) in language modeling and work with the TLDR dataset for a summarization task. Next, we get familiar with Quantization and Instruct Tuning techniques, focusing on Efficient Finetuning of Quantized LLMs (QLoRA). Finally, we delve into the concept of evaluating text generation using BERTScore to compare sentence similarity.

### Reinforcement Learning from Human Feedback (RLHF)

In this part of the project, we implement RLHF for a summarization task using `trlX`. We begin by fine-tuning a pre-trained transformer model on our summarization dataset to create a supervised fine-tuned model (SFT). Next, we train a reward model (RM), initialized from the SFT model, which outputs a scalar value representing the reward that indicates the preferability of a summary. Finally, we use the RM to fine-tune the SFT model via Proximal Policy Optimization (PPO), aligning the SFT model with human preferences.

### Quantization and Instruct Tuning

Quantization is a technique used to reduce the precision of neural network weights and activations, typically from floating-point to lower-bit representations, such as 8-bit or 4-bit integers. The primary goal of quantization is to reduce the memory footprint and computational requirements of deep learning models, allowing larger models to be loaded into available memory and speeding up the inference process. In this project, we use the QLoRA method to examine the effects of quantization on speeding up inference. The notebook covers the entire process, including Hugging Face login, model loading, dataset loading, fine-tuning, and evaluating the performance of the fine-tuned model. Additionally, we explore the effects of Instruct Tuning using the `Mistral-7B-Instruct` model for interactive use cases, with simple experiments provided in the notebook.

### Evaluation

One method of evaluating text generation is by comparing generated text using a language model. In this part of the assignment, we use `BERTScore` to compare the similarity of sentences. We employ a more modern model, `DeBERTa`, for a simple task provided in the notebook, and evaluate its results using both the official `BERTScore` implementation and our own implementation, comparing the outcomes of both approaches.





