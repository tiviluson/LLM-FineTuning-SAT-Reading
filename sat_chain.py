from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()


LLAMA3_SYSTEM_PROMPT = (
    "You are a helpful AI assistant developed by Meta. Respond safely and accurately."
)

template = """Read the passage and answer the question.

### Passage:
{passage}

### Question:
{question}

### Choices:
{choices}

Respond with ONLY the letter and full text of the correct answer."""
prompt = PromptTemplate(
    input_variables=["passage", "question", "choices"],
    template=template,
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY", "empty"),
    openai_api_base=os.getenv("OPENAI_API_BASE_URL", "http://localhost:8000/v1"),
    model="sat-lora",
)


class OutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        return text.split("\n")[-1]


chain = (
    prompt
    | RunnableLambda(
        lambda preprocessed_string: [
            SystemMessage(content=LLAMA3_SYSTEM_PROMPT),
            HumanMessage(content=preprocessed_string.text),
        ]
    ).with_config(run_name="ConstructMessages")
    | llm.with_config(
        run_name="GenerateResponse",
    )
    | OutputParser().with_config(run_name="ParseOutput")
)
