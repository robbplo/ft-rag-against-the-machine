from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
def main():
    loader = DirectoryLoader(
        path="./data/vllm-0.10.1",
        glob="**/[!.]*.py",
    )
    glob = Path.glob(Path("./data/vllm-0.10.1"), "**/[!.]*.py")
    print(len(list(glob)))
    documents: list[Document] = []
    for path in glob:
        content = ""
        with open(path) as file:
            content = file.read()
        print(len(content))
        document = Document(page_content=content)
        documents.append(document)



    # i = 0
    # for document in loader.lazy_load():
    #     print(len(document.page_content))
    #     i +=1
    #     if i == 10:
    #         break
    # loader.loader_kwargs



if __name__ == "__main__":
    main()
