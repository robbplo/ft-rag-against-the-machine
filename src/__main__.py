from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

def main():

    # Load documents
    glob = Path.glob(Path("./data/vllm-0.10.1"), "**/[!.]*.py")
    documents: list[Document] = []
    print("loading documents")
    for path in glob:
        content = ""
        with open(path) as file:
            content = file.read()
        document = Document(page_content=content)
        document.metadata["path"] = path
        documents.append(document)


    # Split text
    print("splitting documents")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    split_documents = splitter.split_documents(documents)

    # Retrieval
    retriever = BM25Retriever.from_documents(split_documents)
    retriever.k = 5
    question = "What is the default lora_int_id value when lora_request is None in vLLM's sequence?"
    results = retriever.invoke(question)
    # print(results[0])
    print(len(documents))


    # Generation
    print("setting up generation")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen3-0.6B",
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate.from_template(
        """# vLLM code expert
        ## Role
        You are an expert on the vLLM codebase. You answer questions from users who are trying to
        work with the repository. Your answers will be based on the given code samples. Answer in a
        technical manner.

        ## Code samples
        {context}

        ## Question
        {question}

        ## Answer
        """
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # answer = rag_chain.invoke(question)

    print(question)
    stream = rag_chain.stream(question)
    for chunk in stream:
        print(chunk, end="", flush=True)




    # i = 0
    # for document in loader.lazy_load():
    #     print(len(document.page_content))
    #     i +=1
    #     if i == 10:
    #         break
    # loader.loader_kwargs



if __name__ == "__main__":
    main()
