import os
import gradio as gr
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain_anthropic import ChatAnthropic

API_KEY = os.getenv('CLAUDE_API_KEY')

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.5, max_tokens=8192, anthropic_api_key=API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

vector_store = None


def process_file(file_path):
    _, ext = os.path.splitext(file_path)
    try:
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif ext.lower() == '.docx':
            with open(file_path, 'rb') as file:
                content = file.read()
                text = content.decode('utf-8', errors='ignore')
        elif ext.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = '\n'.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        else:
            print(f"Unsupported file type: {ext}")
            return None

        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None


def process_files(file_list, progress=gr.Progress()):
    global vector_store
    documents = []
    total_files = len(file_list)

    for i, file in enumerate(file_list):
        progress((i + 1) / total_files, f"Processing file {i + 1} of {total_files}")
        if file.name.lower().endswith(('.txt', '.docx', '.pdf')):
            docs = process_file(file.name)
            if docs:
                documents.extend(docs)

    if not documents:
        return "No documents were successfully processed. Please check your files and try again."

    progress(0.5, "Splitting text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    progress(0.7, "Creating embeddings")
    vector_store = FAISS.from_documents(texts, embeddings)

    progress(0.9, "Saving vector store")
    vector_store.save_local("faiss_index")

    progress(1.0, "Completed")
    return f"Embedding process completed and database created. Processed {len(documents)} files. You can now start chatting!"


def chat(message, history):
    global vector_store
    if vector_store is None:
        return "Please load documents or an existing index first."

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        memory=memory
    )

    result = qa_chain.invoke({"question": message, "chat_history": history})
    return result['answer']


def reset_chat():
    global memory
    memory.clear()
    return []


with gr.Blocks() as demo:
    gr.Markdown("# Document-based Chatbot")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Select Source Knowledge Documents", file_count="multiple", file_types=[".pdf", ".docx", ".txt"])
            process_button = gr.Button("Process Files")

    output = gr.Textbox(label="Processing Output")

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    send = gr.Button("Send")
    clear = gr.Button("Clear")


    def process_selected_files(files):
        if files:
            return process_files(files)
        else:
            return "No files selected. Please select files and try again."


    process_button.click(process_selected_files, file_input, output)


    def respond(message, chat_history):
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(reset_chat, None, chatbot)

if __name__ == "__main__":
    demo.launch()
