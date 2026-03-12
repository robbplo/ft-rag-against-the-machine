from typing import cast
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline, GenerationConfig

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
        # model="Qwen/Qwen3-0.6B",
        model="Qwen/Qwen3.5-0.8B",
    )
    config: GenerationConfig = cast(GenerationConfig, pipe.generation_config)
    config.max_length = None
    config.max_new_tokens = 1024
    config.do_sample = False
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate.from_template(
        """<|im_start|>system
You are a vLLM codebase expert answering questions using only the provided code samples.
Prioritize correctness over completeness.

Rules:
- Base every factual claim on the retrieved code context.
- If the answer is not fully supported by the code samples, say that explicitly and describe what is uncertain.
- Do not invent APIs, behaviors, defaults, control flow, or implementation details.
- Start with the direct answer in the first sentence, then give a brief code-based explanation.
- For code questions, trace the relevant logic step by step using the exact identifiers from the code.
- When a value depends on a condition or branch, explain the condition precisely, including whether conditions are AND, OR, or fallback checks.
- If the question asks for a default, distinguish between:
  1. a signature or field default,
  2. a fallback used when a value is None or missing,
  3. a constant passed into another call.
- For default-value questions, return the concrete value exactly as written in code when possible.
- For "what happens when" questions, describe the exact branch outcome: returned value, raised exception, assertion, mutation, or skipped behavior.
- For condition questions, enumerate every required condition instead of summarizing loosely.
- For supported values, enums, types, or registered names, list all items present in the retrieved code and do not omit values.
- For shape, dtype, annotation, and parameter-type questions, report the exact structure and type information shown in code.
- For error or assertion questions, name the exact exception or assertion and the trigger condition.
- If the code uses attribute access with a fallback such as getattr(..., default) or an if x is None branch, answer with the effective value produced by that path.
- Keep the answer technical, concise, and focused on the repository behavior shown in the context.
- Do not mention information outside the provided code samples.

Code samples:
{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
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
