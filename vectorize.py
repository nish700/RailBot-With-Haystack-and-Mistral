from haystack.dataclasses.document import Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentSplitter

import os

print("===Import Successfull===")

doc_path = "data/"

if __name__=="__main__":

    document_store = QdrantDocumentStore(
        recreate_index=True,
        return_embedding=True,
        wait_result_from_api=True
    )

    print("====Qdrant Started=======")

    converter = PyPDFToDocument()

    files_list = os.listdir(doc_path)

    for file in files_list:
        print("file name is:", file)

        document_path = [os.path.join(doc_path, file)]

        print("===document path:", document_path)

        output = converter.run(sources = document_path)

        docs = output['documents']

        print("======================document processed ==================")

        doc_pdf = docs[0].content

        clean_document = DocumentCleaner(
            remove_empty_lines= True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False
        )

        cleaned_doc = clean_document.run(
            [Document(content = doc_pdf)]
        )

        print("=============document cleaning done============")

        document_splitter = DocumentSplitter(
            split_by="word",
            split_length=512,
            # split_overlap=0
        )

        processed_doc = document_splitter.run(
            [cleaned_doc['documents'][0]]
        )

        print("=======document splitting done======",len(processed_doc))

        for count, docs in enumerate(processed_doc['documents']):
            print("count of document is:", count, "\n",
                  docs)

        print("================document store===========", document_store)

        document_embedder = SentenceTransformersDocumentEmbedder(
            model = "sentence-transformers/all-mpnet-base-v2"
        )

        document_embedder.warm_up()

        document_with_embedding = document_embedder.run(processed_doc['documents'])

        print("======embedding done===========")

        document_store.write_documents(
                document_with_embedding['documents']
        )

        print("=====No. of document in document store: ",document_store.count_documents())
        #
        print("====Embeddings Update Done.....")