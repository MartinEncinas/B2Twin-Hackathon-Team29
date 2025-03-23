import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from tqdm import tqdm  # For progress bars

WORKING_DIR = "/Users/xin/Documents/Documents/hack_az/data"
os.makedirs(WORKING_DIR, exist_ok=True)

def create_and_save_vectorstore():
    print("🚀 Initializing Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="eden")
    print("✅ Ollama initialized")

    try:
        print("📚 Loading text data...")
        # Check if files exist
        files = ["Desert_Temp_RH_FEB-2025.csv", "Desert_CO2_FEB-2025.csv"]
        for file in files:
            if not os.path.exists(file):
                print(f"❌ Error: {file} not found!")
                return

        # Read CSV files using pandas
        print("📊 Reading CSV files...")
        temp_df = pd.read_csv("Desert_Temp_RH_FEB-2025.csv")
        co2_df = pd.read_csv("Desert_CO2_FEB-2025.csv")
        print(f"✅ Loaded {len(temp_df)} temperature records and {len(co2_df)} CO2 records")

        # Convert DataFrames to documents
        print("🔄 Converting data to documents...")
        temp_docs = [{"page_content": row.to_string()} for _, row in tqdm(temp_df.iterrows(), desc="Processing temperature data")]
        co2_docs = [{"page_content": row.to_string()} for _, row in tqdm(co2_df.iterrows(), desc="Processing CO2 data")]
        
        # Create vector store
        print("🔍 Creating vector embeddings (this might take a while)...")
        documents = temp_docs + co2_docs
        vectorstore = FAISS.from_texts(
            texts=[doc["page_content"] for doc in documents],
            embedding=embeddings
        )
        
        print("✅ Vector database created successfully!")
        print(f"📊 Total records processed: {len(documents)}")
        
        # Add this test query
        print("\n🔍 Testing search...")
        results = vectorstore.similarity_search("What is the temperature?", k=1)
        print("Sample result:", results[0].page_content)
        
        # Save the vectorstore
        save_path = os.path.join(WORKING_DIR, "vectorstore")
        vectorstore.save_local(save_path)
        print(f"💾 Vector database saved to {save_path}")
        
        return vectorstore

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

def load_vectorstore():
    """Load the saved vectorstore"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    save_path = os.path.join(WORKING_DIR, "vectorstore")
    if os.path.exists(save_path):
        return FAISS.load_local(save_path, embeddings)
    else:
        return create_and_save_vectorstore()

def query_data(question: str, vectorstore=None, k=3):
    """Query the vector database"""
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    results = vectorstore.similarity_search(question, k=k)
    return [result.page_content for result in results]

# Example usage
if __name__ == "__main__":
    # Create and save the vectorstore (only need to do this once)
    vectorstore = create_and_save_vectorstore()
    
    # Example queries
    questions = [
        "What is the highest temperature recorded?",
        "What are the CO2 levels at midnight?",
        "Show me temperature readings above 90 degrees"
    ]
    
    for question in questions:
        print(f"\n🔍 Query: {question}")
        answers = query_data(question, vectorstore)
        print("Results:")
        for i, answer in enumerate(answers, 1):
            print(f"\n{i}. {answer}")