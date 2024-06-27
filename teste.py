from llama_cpp import Llama
llm = Llama(model_path="models/llama-2-7b-chat.ggmlv3.q4_0.bin")

from ctransformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained(
    'models/llama-2-7b-chat.ggmlv3.q4_0.bin',
     model_type='llama')

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

template = """Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type='llama')
llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run("Por que o céu é azul?")

print(response)