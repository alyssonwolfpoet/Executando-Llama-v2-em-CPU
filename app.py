from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import chainlit as cl

@cl.on_chat_start
def load():
  template = """Responda a essa pergunta do usuário: {question}.
             """
  prompt = PromptTemplate(template=template, input_variables=["question"])

  llm = CTransformers(model="models/llama-2–7b-chat.ggmlv3.q4_0.bin",
  model_type='llama')

  llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

  cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
  llm_chain = cl.user_session.get("llm_chain")
  res = await cl.make_async(llm_chain)(message,
                                    callbacks=[cl.LangchainCallbackHandler()]
                                   )
  await cl.Message(res['text']).send()
  return llm_chain