from flask import Flask, request, jsonify
import os
import pinecone
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Access environment variables for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Ensure the API keys are set
if not openai_api_key or not pinecone_api_key:
    raise Exception("API keys not set in environment variables")

# Set up OpenAI and Pinecone API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

# Load data, split text, create embeddings, and set up vectorstore
#loader = PyPDFLoader("swarmdata.pdf")
#docs = loader.load_and_split()

#text_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=1200,
#    chunk_overlap=200,
#    length_function=len,
#)

#docs_chunks = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

index_name = "swarm"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Create a question answering chain
llm = OpenAI()
qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# Define a route to handle the incoming queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json  # Assuming the query is sent in JSON format in the request body
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Perform the query and return the result
    result = qa_with_sources({"query": query})
    response = {"result": result["result"]}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
