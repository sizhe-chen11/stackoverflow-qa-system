from langchain.docstore.document import Document

class StackOverflowDataLoader:
    def __init__(self):
        # Simple test data - no downloads needed
        self.test_data = [
            {
                "question": "How do I handle exceptions in Python?",
                "answer": "Use try-except blocks. Put code that might fail in the try block, and handle errors in the except block."
            },
            {
                "question": "What is a Python list?", 
                "answer": "A list is a collection of items in Python. You create it with square brackets like [1, 2, 3]."
            },
            {
                "question": "How to create a function in Python?",
                "answer": "Use the def keyword followed by function name and parentheses. Example: def my_function(): return 'hello'"
            },
            {
                "question": "What is a for loop in Python?",
                "answer": "A for loop iterates over a sequence. Example: for item in [1,2,3]: print(item)"
            },
            {
                "question": "How to install packages in Python?",
                "answer": "Use pip install package_name in the command line to install Python packages."
            }
        ]
        
    def load_data(self, max_samples=5):
        print(f"Using {len(self.test_data)} test questions")
        return self.test_data
        
    def convert_to_documents(self, dataset):
        documents = []
        for i, item in enumerate(dataset):
            content = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
            doc = Document(
                page_content=content,
                metadata={'source': 'test_data', 'index': i}
            )
            documents.append(doc)
        print(f"Created {len(documents)} documents")
        return documents