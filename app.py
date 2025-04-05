from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from dotenv import load_dotenv
import emoji
import os
from itertools import zip_longest

# Import Google's genai and types directly
from google import genai
from google.genai import types

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),  # Or GEMINI_API_KEY depending on your .env
)
# Initialize the Gemini API client - using your reference approach
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),  # or GEMINI_API_KEY depending on your .env file
)

st.title(f"Career Advisor Chatbot {emoji.emojize(':robot:')}")

global vectors
# Define your directory containing PDF files here
pdf_dir = 'pdf'

if "pdf_texts" not in st.session_state:
    temp_pdf_texts = []
    with st.spinner("Creating a Database..."):
        for file in os.listdir(pdf_dir):
            if file.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                documents = loader.load()
                text = " ".join([doc.page_content for doc in documents])
                temp_pdf_texts.append(text)
        st.session_state["pdf_texts"] = temp_pdf_texts
        pdf_list = list(st.session_state["pdf_texts"])
        pdfDatabase = " ".join(pdf_list)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(pdfDatabase)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if "vectors" not in st.session_state: 
            vectors = FAISS.from_texts(chunks, embeddings)
            st.session_state["vectors"] = vectors
    st.success("Database creation completed!")

def get_response(history, user_message):
    # Get relevant documents from vector store
    docs = st.session_state["vectors"].similarity_search(user_message)
    doc_text = " ".join([doc.page_content for doc in docs])
    
    # Get web search results
    params = {
        "engine": "bing",
        "gl": "us",
        "hl": "en",
    }
    search = SerpAPIWrapper(params=params)
    web_knowledge = search.run(user_message)
    
    # Create prompt content
    prompt = f"""The following is a friendly conversation between a human and a Career Advisor. The Advisor guides the user regarding jobs, interests and other domain selection decisions.
    It follows the previous conversation to do so.

    Relevant pieces of previous conversation:
    {history}

    Useful information from career guidance books:
    {doc_text}

    Useful information about career guidance from Web:
    {web_knowledge}

    Current conversation:
    Human: {user_message}
    Career Expert:"""
    
    # Create content structure following your reference
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    # Setup generation config
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    
    # Use the model directly with the client - Update the model name as needed
    model = "gemini-2.0-flash"  # or use "gemini-2.5-pro-preview-03-25" if available to you
    
    # For Streamlit, we need to collect the full response rather than streaming directly
    full_response = ""
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            full_response += chunk.text
    
    return full_response

# Function to get conversation history
def get_history(history_list):
    history = ''
    for message in history_list:
        if message['role'] == 'user':
            history = history + 'input ' + message['content'] + '\n'
        elif message['role'] == 'assistant':
            history = history + 'output ' + message['content'] + '\n'
    
    return history

# Streamlit UI
def get_text():
    input_text = st.sidebar.text_input("You: ", "Hello, how are you?", key="input")
    if st.sidebar.button('Send'):
        return input_text
    return None

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

user_input = get_text()

if user_input:
    user_history = list(st.session_state["past"])
    bot_history = list(st.session_state["generated"])

    combined_history = []
    for user_msg, bot_msg in zip_longest(user_history, bot_history):
        if user_msg is not None:
            combined_history.append({'role': 'user', 'content': user_msg})
        if bot_msg is not None:
            combined_history.append({'role': 'assistant', 'content': bot_msg})

    formatted_history = get_history(combined_history)

    output = get_response(formatted_history, user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Chat History", expanded=True):
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            st.markdown(emoji.emojize(f":speech_balloon: **User {str(i)}**: {st.session_state['past'][i]}"))
            st.markdown(emoji.emojize(f":robot: **Assistant {str(i)}**: {st.session_state['generated'][i]}"))