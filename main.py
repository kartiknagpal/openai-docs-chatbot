from llama_index import SimpleDirectoryReader, ServiceContext, StorageContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper, load_index_from_storage
from langchain import OpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = 'YOUR-API-KEY'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2048
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index.storage_context.persist()

    return index

def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    #print(response)
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="My AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)