
from haystack_integrations.components.retrievers.qdrant.retriever import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder


from haystack import Pipeline
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from fastapi import FastAPI,Request, Form, Response

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

import json



print("Import Successfully")

app = FastAPI()

templates = Jinja2Templates(directory='templates')

app.mount("/static", StaticFiles(directory='static'), name='static')


def get_result(query):

    document_store = QdrantDocumentStore(
        recreate_index=False,
        return_embedding=True,
        wait_result_from_api=True
    )

    text_embedder = SentenceTransformersTextEmbedder(
        model = "sentence-transformers/all-mpnet-base-v2"
    )

    prompt_template = """
        Given these documents, answer the user query. Make the answer to the point and do not make it yourself.
        Do not hallucinate. If you can't find the answer in the document , just say Unable to find the response\nDocuments:
        {% for doc in documents %}
            {{ doc.content}}
        {% endfor %}
        
        \nQuery:{{query}}
        \nAnswer:
        """

    generator = LlamaCppGenerator(
        model_path = 'model/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        # n_ctx = 8000,
        n_batch = 64,
        model_kwargs = {
            'n_gpu_layers': 2
        },
        generation_kwargs = {
                'max_tokens' : 256,
                'temperature': 0
        }
    )

    retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k = 1,
        return_embedding = True
        )


    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=text_embedder, name="text_embedder")
    rag_pipeline.add_component(instance= retriever, name="embedding_retriever")
    rag_pipeline.add_component(instance= PromptBuilder(template=prompt_template), name="prompt_builder")
    rag_pipeline.add_component(instance=generator, name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    rag_pipeline.connect("text_embedder", "embedding_retriever")
    rag_pipeline.connect("embedding_retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies","answer_builder.replies")
    rag_pipeline.connect("embedding_retriever", "answer_builder.documents")

    print("====rag_pipeline:",rag_pipeline)

    json_response = rag_pipeline.run({
        "text_embedder":{"text" : query},
        "answer_builder" : {"query" : query},
        "prompt_builder" : {"query" : query},

    })


    print("======Json Response is:======", json_response)
    generated_answer = json_response['llm']['meta'][0]['choices'][0]['text']
    doc_ref = json_response['answer_builder']['answers'][0].documents[0].content


    print("answers:", generated_answer, type(generated_answer))
    print("===documents:", doc_ref)


    relevant_documents = doc_ref

    return generated_answer, relevant_documents


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.post("/get_answer")
async def get_answer(request: Request, question: str= Form(...)):
    print("==============================",question)
    # question="OFFENCES AGAINST THE STATE comes under which chapter?"
    answer, relevant_doc = get_result(question)
    response_data = jsonable_encoder(json.dumps({"answer": answer, "relevantDocs": relevant_doc}))
    res = Response(response_data)

    return res
