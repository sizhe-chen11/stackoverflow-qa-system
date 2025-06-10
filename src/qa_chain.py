from transformers import AutoTokenizer, AutoModel
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

class SimpleChatGLM(LLM):
    def __init__(self):
        print("Loading ChatGLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        print("ChatGLM model loaded!")
    
    @property
    def _llm_type(self):
        return "chatglm"
    
    def _call(self, prompt, stop=None, **kwargs):
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

class QAChain:
    def __init__(self):
        self.llm = SimpleChatGLM()
        
    def create_qa_chain(self, vectorstore):
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    
    def ask_question(self, qa_chain, question):
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }