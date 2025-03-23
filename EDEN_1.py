import gradio as gr
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

WORKING_DIR = "/Users/xin/Documents/Documents/hack_az/data"
VECTORSTORE_PATH = os.path.join(WORKING_DIR, "vectorstore")

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="qwen2.5:7b")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
llm = OllamaLLM(model="qwen2.5:7b")

def query_data(question: str, k=5):
    """Query the loaded vector store with a given question."""
    results = vectorstore.similarity_search(question, k=k)
    retrieved_docs = [result.page_content for result in results]
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {question}\nContext:\n{context}\nAnswer:"
    response = llm.generate(prompts=[prompt])
    answer = response.generations[0][0].text.strip()  # Extract the text from the response
    return answer

def chatbot_interface(user_input, history):
    """Handle each chat message."""
    history = history or []
    answer = query_data(user_input)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbox with Qwen")
    
    chat_history = gr.State([])
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(type='messages')  # Set type to 'messages'
            user_msg = gr.Textbox(show_label=False, placeholder="Ask about the CSV data...")
            send_btn = gr.Button("Send")

    send_btn.click(
        fn=chatbot_interface,
        inputs=[user_msg, chat_history],
        outputs=[chatbot, chat_history]
    )

demo.launch(server_name="0.0.0.0", server_port=7862)