# RailBot-With-Haystack-and-Mistral
Creating a Railway Chatbot with the aid of Haystack 2.0 framework and using Mistral 7B Instruct LLM for chat Question Answering

Haystack is an open-source Python framework developed by deepset for building custom applications with large language models (LLMs). It allows you to quickly experiment with the latest NLP models while maintaining flexibility and ease of use. Haystack 2.0 allows to integrate open source model to the pipeline with the LLamaCppGenerator.

Open Source Document store - Qdrant has been used to store the text embedding. Qdrant has the adantage of ease of scalabitlity.

The Qdrant Document Store is hosted on the Docker Container, so that it can be easily used across various environments.

Sentence Trsnaformer has been used to vectorize the document and generate the embeddings.

The following Libraries used for developing the chatbot:

- haystack-ai
- fastapi
- uvicorn
- sentence-transformers
- docker
- qdrant
- llamacpp

