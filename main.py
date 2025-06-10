from src.data_loader import StackOverflowDataLoader
from src.processor import DocumentProcessor

def main():
    print("🚀 Simple Search System Test...")
    
    # Step 1: Load test data
    print("Loading test data...")
    data_loader = StackOverflowDataLoader()
    dataset = data_loader.load_data()
    documents = data_loader.convert_to_documents(dataset)
    
    # Step 2: Create vectorstore  
    print("Creating vectorstore...")
    processor = DocumentProcessor()
    texts = processor.split_documents(documents)
    vectorstore = processor.create_vectorstore(texts)
    print("Vectorstore created!")
    
    # Step 3: Interactive search
    print("\n=== Interactive Search System ===")
    print("Ask questions and get relevant documents (no AI generation)")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Search for relevant documents
        results = vectorstore.similarity_search(question, k=2)
        
        print(f"\n🔍 Found {len(results)} relevant results:")
        for i, doc in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            print(doc.page_content)
            print("-" * 50)

if __name__ == "__main__":
    main()