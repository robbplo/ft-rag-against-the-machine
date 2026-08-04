*This project has been created as part of the 42 curriculum by rploeger.*

# RAG Against the Machine

## Description

This project is a local Retrieval-Augmented Generation (RAG) system for the
vLLM codebase. It indexes source code and documentation, retrieves the most
relevant passages for a question, and asks `Qwen/Qwen3-0.6B` through the
Transformers text-generation pipeline to produce a grounded answer.

## System architecture

```text
vLLM files -> chunking -> BM25 index -> top-k passages -> Qwen -> answer
                                      \-> recall@k evaluation
```

- `SourceLoader` discovers supported code, documentation, and text files.
- The index stores each chunk with its exact file path and character offsets.
- The retriever ranks chunks for either one question or a JSON dataset.
- The answer generator fits retrieved context into the model token budget.
- The evaluator compares retrieved ranges with known sources using overlap.

## Technical approach

### Chunking strategy

Python code is split with language-aware separators. Markdown uses
heading-aware separators, while plain-text files use recursive text splitting.
Chunks default to 2,000 characters, overlap by 10%, and preserve their original
character positions through splitter-provided start-index metadata.

### Retrieval method

The default index uses BM25 with English stemming. The corpus and query use the
same tokenizer. BM25 rewards chunks containing important query terms, while
normalizing for term frequency and chunk length; chunks with the highest BM25
scores are returned first.

The retrieval layer can combine multiple indexes through weighted Reciprocal
Rank Fusion (RRF). Each strategy retrieves `4 * k` candidates. A candidate at
zero-based rank `r` contributes `weight / (60 + r + 1)` to its combined score,
and duplicate file/character ranges are merged before the final top-k is
selected. Currently only BM25 is enabled, with weight `0.925`, which keeps the
mandatory pipeline fast and CPU-friendly while leaving the same interface
available for optional semantic retrieval.

### Design decisions

Exact paths and character offsets are retained because evaluation matches both
the file and overlapping source range. Separate splitters keep code structures
and document sections more meaningful than fixed cuts. A strict 2,000-character
limit prevents invalid evaluator output, while overlap reduces information loss
at chunk boundaries. Model context is tokenized and truncated to the model's
supported input length to avoid exceeding Qwen's input capacity.

### Challenges faced

- **Preserving exact source locations:** The same text can occur more than once
  in a file, so searching the original file for a chunk can produce the wrong
  character offset. The splitters therefore record each chunk's start index as
  metadata, and the end index is calculated from that exact position.
- **Keeping useful context at chunk boundaries:** Hard fixed cuts can separate a
  definition from its explanation. Language-aware separators are used where
  possible, with recursive fallback splitting and 10% overlap between chunks.
- **Balancing quality and CPU performance:** Lexical retrieval is fast and
  performs well for exact identifiers, but it can miss synonymous wording.
  BM25 remains the mandatory default so the time and recall requirements are
  met; the weighted RRF abstraction allows semantic retrieval to be added
  without changing the CLI or output format.
- **Fitting evidence into the model context:** Retrieved chunks are joined,
  tokenized, and truncated to the tokenizer's supported input length before the
  prompt is sent to Qwen.

## Instructions

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for this project.
Extract the supplied vLLM archive to `data/raw/vllm-0.10.1`, then run:

```bash
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How is prefix caching configured?" --k 5
uv run python -m src answer "How is prefix caching configured?" --k 5
```

Dataset search, evaluation, and answer generation:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Useful Make targets are `install`, `run`, `debug`, `clean`, and `lint`.

## Performance analysis

Using the included public results, the local evaluator reports:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Documentation (100 questions) | 61.0% | 76.0% | 84.0% | 89.0% |
| Code (99 questions) | 33.3% | 44.4% | 54.5% | 58.6% |

Both Recall@5 values meet the subject thresholds of 80% for documentation and
50% for code. Lexical retrieval is fast and lightweight, but can miss passages
when a question uses vocabulary different from the source.

A local CPU benchmark on an arm64 development machine produced the following
wall-clock timings. Dependencies were already installed, and the retrieval test
includes process startup, loading the persisted index, searching, and writing
JSON output. Exact timings vary by machine.

| Operation | Workload | Measured time | Subject limit |
| --- | --- | ---: | ---: |
| Indexing | 1,969 files / 14,638 chunks | 3.37 s | 300 s |
| Retrieval | 199 questions at `k=10` | 5.14 s | 90 s |

The retrieval timing is the sum of separate 100- and 99-question dataset runs,
so index loading and process startup are included twice. This makes the comparison
more conservative than processing all 200 questions in one invocation.

## Resources

- [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- [BM25 overview](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Pydantic documentation](https://docs.pydantic.dev/)

### AI usage

AI was used to compare the implementation with the subject requirements,
discuss indexing and retrieval design choices, review code, and help draft this
documentation. Technical descriptions were checked against the implementation,
and the recall and timing figures were produced by running the local pipeline
against the public datasets rather than accepting generated estimates.
