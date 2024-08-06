import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from firebase_admin import credentials
from firebase_admin import auth
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
import os
import json
import requests
import tempfile
from functools import partial
import datetime
import pytz
import mimetypes
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

### Functions: Start ###

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

@st.dialog("Google Consent Authentication Link")
def google_oauth_link(flow):
    auth_url, _ = flow.authorization_url(redirect_uris=st.secrets["web"]["redirect_uris"], prompt='consent')
    st.write("Please go to this URL and authorize access:")
    st.markdown(f"[Sign in with Google]({auth_url})", unsafe_allow_html=True)
    code = st.text_input("Enter the authorization code:")
    return code

def fetch_token_data():
    """Fetch the token data from Firestore."""
    try:
        # Fetch the first document from the collection
        token_ref = st.session_state.db.collection('Token').limit(1)
        token_docs = token_ref.get()
        
        # Check if the collection is not empty
        if not token_docs:
            st.error("No token document found in Firestore.")
            return None, None
        
        # Get the first document
        token_doc = None
        doc_id = None
        for doc in token_docs:
            token_doc = doc.to_dict()
            doc_id = doc.id
            break

        if not token_doc:
            st.error("No token document found in Firestore.")
            return None, None

        return token_doc, doc_id
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None, None

def load_creds():
    token_doc, doc_id = fetch_token_data()

    if token_doc:
        account = token_doc.get("account")
        client_id = token_doc.get("client_id")
        client_secret = token_doc.get("client_secret")
        expiry = token_doc.get("expiry")
        refresh_token = token_doc.get("refresh_token")
        scopes = token_doc.get("scopes")
        token = token_doc.get("token")
        token_uri = token_doc.get("token_uri")
        universe_domain = token_doc.get("universe_domain")

        st.session_state['account'] = account
        st.session_state['client_id'] = client_id
        st.session_state['client_secret'] = client_secret
        st.session_state['expiry'] = expiry
        st.session_state['refresh_token'] = refresh_token
        st.session_state['scopes'] = scopes
        st.session_state['token'] = token
        st.session_state['token_uri'] = token_uri
        st.session_state['universe_domain'] = universe_domain

        # Create a temporary directory and dump the token info into a JSON file
        temp_dir = tempfile.mkdtemp()
        token_file_path = os.path.join(temp_dir, 'token.json')
        with open(token_file_path, 'w') as token_file:
            json.dump(token_doc, token_file)

        creds = Credentials.from_authorized_user_file(token_file_path, scopes)

        if creds.expired:
            token_doc, _ = fetch_token_data()
            if token_doc:
                new_refresh_token = token_doc.get("refresh_token")
                if creds.refresh_token and creds.refresh_token == new_refresh_token:
                    st.toast("Refreshing token...")
                    creds.refresh(Request())
                    new_token_data = {
                        "account": account,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "expiry": creds.expiry.isoformat() if creds.expiry else expiry,
                        "refresh_token": creds.refresh_token,
                        "scopes": scopes,
                        "token": creds.token,
                        "token_uri": token_uri,
                        "universe_domain": universe_domain
                    }
                    
                    st.session_state.db.collection('Token').document(doc_id).set(new_token_data)
                    st.session_state.update(new_token_data)
                    with open(token_file_path, 'w') as token_file:
                        json.dump(new_token_data, token_file)
                else:
                    st.error("Refresh token mismatch or missing. Please log in again.")
                    return None
            else:
                st.error("Failed to re-fetch token data from Firestore.")
                return None
    else:
        return None

    return creds

@st.cache_resource
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def download_file_to_temp(url):
    storage_client = storage.Client.from_service_account_info(st.session_state["connext_chatbot_admin_credentials"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
    temp_dir = tempfile.mkdtemp()

    parsed_url = urlparse(url)
    file_name = os.path.basename(unquote(parsed_url.path))

    blob = bucket.blob(file_name)
    temp_file_path = os.path.join(temp_dir, file_name)
    blob.download_to_filename(temp_file_path)

    return temp_file_path, file_name

def extract_and_parse_json(text):
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False

    json_str = text[start_index:end_index + 1]
    try:
        parsed_json = json.loads(json_str)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False
    
def is_expected_json_content(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False
    
    required_keys = ["Is_Answer_In_Context", "Answer"]
    return all(key in data for key in required_keys)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        text += extract_text(pdf)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_generative_model(response_mime_type="text/plain"):
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "max_output_tokens": 8192,
        "response_mime_type": response_mime_type
    }

    if st.session_state["oauth_creds"] is not None:
        genai.configure(credentials=st.session_state["oauth_creds"])
    else:
        st.session_state["oauth_creds"] = load_creds()
        genai.configure(credentials=st.session_state["oauth_creds"])

    model_name = 'tunedModels/connext-wide-chatbot-ddal5ox9d38h' if response_mime_type == "text/plain" else "gemini-1.5-flash"
    return genai.GenerativeModel(model_name, generation_config=generation_config)

def generate_response(question, context, fine_tuned_knowledge=False):
    prompt = (f"""
    Based on your base or fine-tuned knowledge, can you answer the following question?

    --------------------

    Question:
    {question}

    --------------------

    Answer:
    """ if fine_tuned_knowledge else f"""
    Answer the question below as detailed as possible from the provided context below, make sure to provide all the details but if the answer is not in
    provided context. Try not to make up an answer just for the sake of answering a question.

    --------------------
    Context:
    {context}

    --------------------

    Question:
    {question}

    Provide your answer in a json format following the structure below:
    {{
        "Is_Answer_In_Context": <boolean>,
        "Answer": <answer (string)>,
    }}
    """)

    model = get_generative_model("text/plain" if fine_tuned_knowledge else "application/json")
    return model.generate_content(prompt).text

def try_get_answer(user_question, context="", fine_tuned_knowledge=False):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = generate_response(user_question, context, fine_tuned_knowledge)
            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid and is_expected_json_content(parsed_result):
                return parsed_result
        except Exception as e:
            st.toast(f"Failed to create a response for your query. Error: {str(e)} \nTrying again... Retries left: {max_attempts - attempt - 1}")
    return ""

def user_input(user_question, api_key, chat_history):
    with st.spinner("Processing..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        context = "\n\n--------------------------\n\n".join([f"User: {entry['question']}\nBot: {entry['answer']['Answer']}" for entry in chat_history])
        context += "\n\n--------------------------\n\n"
        context += "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        return try_get_answer(user_question, context)

def app():
    google_ai_api_key = st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(st.session_state["connext_chatbot_admin_credentials"]))

    st.session_state.db = firestore.client()

    col1, col2, col3 = st.columns([3, 4, 3])
    with col1:
        st.write(' ')
    with col2:
        st.image("Connext_Logo.png", width=250)
    with col3:
        st.write(' ')

    st.markdown('## Welcome to :blue[Connext Chatbot] :robot_face:')

    retrievers_ref = st.session_state.db.collection('Retrievers')
    docs = retrievers_ref.stream()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'parsed_result' not in st.session_state:
        st.session_state.parsed_result = {}

    chat_history_placeholder = st.empty()

    def display_chat_history():
        with chat_history_placeholder.container():
            for chat in st.session_state.chat_history:
                st.markdown(f"🧑 **You:** {chat['question']}")
                st.markdown(f"🤖 **Bot:** {chat['answer']['Answer']}")

    display_chat_history()

    user_question = st.text_input("Ask a Question", key="user_question")
    submit_button = st.button("Submit", key="submit_button")
    clear_history_button = st.button("Clear Chat History")

    if clear_history_button:
        st.session_state.chat_history = []
        display_chat_history()

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = {}

    if "selected_retrievers" not in st.session_state:
        st.session_state["selected_retrievers"] = []

    if "answer" not in st.session_state:
        st.session_state["answer"] = ""

    if "request_fine_tuned_answer" not in st.session_state:
        st.session_state["request_fine_tuned_answer"] = False

    if 'fine_tuned_answer_expander_state' not in st.session_state:
        st.session_state.fine_tuned_answer_expander_state = False

    if 'show_fine_tuned_expander' not in st.session_state:
        st.session_state.show_fine_tuned_expander = False

    if submit_button:
        if user_question and google_ai_api_key:
            parsed_result = user_input(user_question, google_ai_api_key, st.session_state.chat_history)
            st.session_state.parsed_result = parsed_result
            if "Answer" in parsed_result:
                st.session_state.chat_history.append({"question": user_question, "answer": parsed_result})
                display_chat_history()
                if "Is_Answer_In_Context" in parsed_result and not parsed_result["Is_Answer_In_Context"]:
                    st.session_state.show_fine_tuned_expander = True
            else:
                st.toast("Failed to get a valid response from the model.")

    display_chat_history()

    if st.session_state.show_fine_tuned_expander:
        with st.expander("Get fine-tuned answer?", expanded=True):
            st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
            col1, col2, _ = st.columns([1, 1, 1])
            with col1:
                if st.button("Yes", key=f"yes_button"):
                    st.session_state.request_fine_tuned_answer = True
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()
            with col2:
                if st.button("No", key=f"no_button"):
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()

    if st.session_state["request_fine_tuned_answer"]:
        fine_tuned_result = try_get_answer(st.session_state.chat_history[-1]['question'], context="", fine_tuned_knowledge=True)
        if fine_tuned_result:
            st.session_state.chat_history[-1]['answer'] = {"Answer": fine_tuned_result.strip()}
            display_chat_history()
        else:
            st.toast("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False

    with st.sidebar:
        st.title("PDF Documents:")
        for idx, doc in enumerate(docs, start=1):
            retriever = doc.to_dict()
            retriever['id'] = doc.id
            retriever_name = retriever['retriever_name']
            retriever_description = retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")
                file_path, file_name = download_file_to_temp(retriever['document'])
                st.markdown(f"_**File Name**_: {file_name}")
                retriever["file_path"] = file_path
                st.session_state["retrievers"][retriever_name] = retriever
        st.title("PDF Document Selection:")
        st.session_state["selected_retrievers"] = st.multiselect("Select Documents", list(st.session_state["retrievers"].keys()))

        if st.button("Submit & Process", key="process_button"):
            if google_ai_api_key:
                with st.spinner("Processing..."):
                    selected_files = [st.session_state["retrievers"][name]["file_path"] for name in st.session_state["selected_retrievers"]]
                    raw_text = get_pdf_text(selected_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, google_ai_api_key)
                    st.success("Done")
            else:
                st.toast("Failed to process the documents", icon="💥")

if __name__ == "__main__":
    app()
