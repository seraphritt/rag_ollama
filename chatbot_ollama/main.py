# python 3.9.18

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain import PromptTemplate


def load_pdf_data(file_path):
    # Creating a PyMuPDFLoader object with file_path
    loader = PyMuPDFLoader(file_path=file_path)

    # loading the PDF file
    docs = loader.load()

    # returning the loaded document
    return docs


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)

    # returning the document chunks
    return chunks


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding  # keep True to compute cosine similarity
        }
    )


# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Saving the model in current directory
    vectorstore.save_local(storing_path)

    # returning the vectorstore
    return vectorstore


template = """
### System:
You are an respectful and honest assistant specialized to answer about University of Brasília. \
Always answer in Portuguese from Brazil. \
All your answers from now on most be in Portuguese. \
All your answers must be strictly related to the context passed for you.

### Context:
{context}

### User:
{question}

### Response:
"""


def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )


def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})

    # Wrapping the text for better output in Jupyter Notebook
    # wrapped_text = textwrap.fill(response['result'], width=100)
    print(f"{response['result']}\n\n")


llm = Ollama(model="llama2", temperature=0.1)

# Loading the Embedding Model
embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

# loading and splitting the documents
docs = load_pdf_data(file_path="guia_calouro_1_2018.pdf")
documents = split_docs(documents=docs)

# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

print(prompt)
# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)

while True:
    get_response(input("Digite sua pergunta: "), chain)
