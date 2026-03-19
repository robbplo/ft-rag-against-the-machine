from src.answer_generator import AnswerGenerator
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

def load_documents() -> list[Document]:
    glob = Path.glob(Path("./data/vllm-0.10.1"), "**/[!.]*.py")
    documents: list[Document] = []
    for path in glob:
        content = ""
        with open(path) as file:
            content = file.read()
        document = Document(page_content=content)
        document.metadata["path"] = path
        documents.append(document)
    return documents

def split_documents(documents) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
    )
    split_documents = splitter.split_documents(documents)
    for document in split_documents:
        start_index = document.metadata.get("start_index")
        if isinstance(start_index, int) and document.page_content:
            document.metadata["end_index"] = start_index + len(document.page_content) - 1
    return split_documents

def answer_question(generator: AnswerGenerator, question: str):
    ouptut = ""
    for chunk in generator.stream(question):
        print(chunk, end="", flush=True)
        ouptut += chunk

    answer = re.sub(r"[\s\S]*<\/think>([\s\S]*)", r"\1", ouptut).strip()
    return answer


def main():
    question = "What activation formats does the fused batched MoE layer return in vLLM?"

    print("loading documents")
    documents = load_documents()

    print("splitting documents")
    split_docs = split_documents(documents)

    print("searching documents")
    retriever = BM25Retriever.from_documents(split_docs)
    retriever.k = 5

    print("setting up generation")
    generator = AnswerGenerator(
        model_id="Qwen/Qwen3-0.6B",
        retriever=retriever
    )

    print("starting answer stream")
    answer = answer_question(generator, question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
