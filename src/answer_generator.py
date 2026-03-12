from typing import cast
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, pipeline, AutoTokenizer

PROMPT_TEMPLATE = """<|im_start|>system
You are a vLLM codebase expert answering questions using only the provided code samples.
Prioritize correctness over completeness.

Rules:
- Base every factual claim on the retrieved code context.
- If the answer is not fully supported by the code samples, say that explicitly and describe what is uncertain.
- Do not invent APIs, behaviors, defaults, control flow, or implementation details.
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
{documents}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

class AnswerGenerator:
    def __init__(self, model_id: str, documents: list[Document]):
        model_config = AutoConfig.from_pretrained(model_id)
        model_config.tie_word_embeddings = False
        model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        pipe = pipeline(
            "text-generation",
            model=model_id,
            # model=model,
            # tokenizer=model_id,
        )
        generation_config: GenerationConfig = cast(GenerationConfig, pipe.generation_config)
        generation_config.max_length = None
        generation_config.max_new_tokens = 1024
        generation_config.do_sample = False
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).partial(
            documents=format_docs(documents)
        )

        self.chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def stream(self, question):
        return self.chain.stream(question)




    # answer = rag_chain.invoke(question)
