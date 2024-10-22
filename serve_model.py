
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
llm = ChatOllama(
    # model="qwen2:1.5b-instruct-fp16",
    model="gemma2:2b-instruct-fp16",
    top_p=0.8,
    temperature=0.3,
    # disable_streaming=True,
    seed=451999,
    num_predict=1024
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a helpful assistant."
        ),
        (
            "human",
            "{prompt}"
        )
    ]
)
chain = prompt_template | llm
prompt = "Hello, how are you?"

result = chain.invoke(
    {
        "prompt": prompt
    }
)

output = result.content

print(output)