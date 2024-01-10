import runhouse as rh

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langserve.client import RemoteRunnable

import chromadb

# Add typing for input
class Question(BaseModel):
    __root__: str

class LangChainApp(rh.Module):
    def __init__(self):
        super().__init__()
        self.chain = None
        self.collection = None
        self.persistent_client = chromadb.PersistentClient()
        
    def load_data(self, url):

        # Load 
        loader = WebBaseLoader(url)
        data = loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        self.collection = self.persistent_client.get_or_create_collection("rag-chroma")
        self.collection.add(documents=all_splits)

    def setup_chain(self):
        vectorstore = Chroma(
            client=self.persistent_client,
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(),
        )

        retriever = vectorstore.as_retriever()

        # RAG prompt
        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # LLM
        model = ChatOpenAI()

        # RAG chain
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | model
            | StrOutputParser()
        )

        self.chain = chain.with_types(input_type=Question)

    def predict(self, prompt):
        return self.chain.invoke(prompt)

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    gpu = rh.ondemand_cluster(name='rh-lang', instance_type='1CPU--1GB')
    env = rh.env(reqs=["langchain-cli", "BeautifulSoup4",
                       "langserve[all]", "langchain-community", "langchain-core", "openai"], name="ragchroma", secrets=["openai"])
    
    remote_langchain_model = LangChainApp().to(gpu, env=env, name="rag-chroma")

    # Load in remote data source(s)
    remote_langchain_model.load_data("https://www.run.house/")

    # Create RAG chain 
    remote_langchain_model.setup_chain()

    # Inference 
    test_output = remote_langchain_model.predict("What is Runhouse?")

    print(test_output)





    
    