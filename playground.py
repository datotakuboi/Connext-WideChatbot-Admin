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
        token_ref = st.session_state.db.collection('Token').limit(1)
        token_docs = token_ref.get()
        
        if not token_docs:
            st.error("No token document found in Firestore.")
            return None, None
        
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

def download_file_to_temp(url):
    # Create a temporary directory
    storage_client = storage.Client.from_service_account_info(st.session_state["connext_chatbot_admin_credentials"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
    temp_dir = tempfile.mkdtemp()

    # Download the file
    response = requests.get(url)
    parsed_url = urlparse(url)
    file_name = os.path.basename(unquote(parsed_url.path))

    blob = bucket.blob(file_name)
    
    # Create the full path with the preferred filename
    temp_file_path = os.path.join(temp_dir, file_name)

    blob.download_to_filename(temp_file_path)

    return temp_file_path, file_name

def extract_and_parse_json(text):
    # Find the first opening and the last closing curly brackets
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False  # Proper JSON structure not found

    # Extract the substring that contains the JSON
    json_str = text[start_index:end_index + 1]

    try:
        # Attempt to parse the JSON
        parsed_json = json.loads(json_str)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False  # JSON parsing failed
    
def is_expected_json_content(json_data):
    try:
        # Try to load the JSON data
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False
    
    required_keys = ["Is_Answer_In_Context", "Answer"]

    if not all(key in data for key in required_keys):
            return False
    
    return True #All checks passed for the specified type

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        extracted_text = extract_text(pdf)
        text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_generative_model(response_mime_type = "text/plain"):
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


    model = genai.GenerativeModel('tunedModels/connext-wide-chatbot-ddal5ox9d38h' ,generation_config=generation_config) if response_mime_type == "text/plain" else genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    print(f"Model selected: {model}")
    return model

def generate_response(question, context, fine_tuned_knowledge = False):

    prompt_using_fine_tune_knowledge = f"""
    Based on your base or fine-tuned knowledge, can you answer the the following question?

    --------------------

    Question:
    {question}

    --------------------

    Answer:

    """
    prompt_with_context = f"""

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
    """

    prompt = prompt_using_fine_tune_knowledge if fine_tuned_knowledge else prompt_with_context
    model = get_generative_model("text/plain" if fine_tuned_knowledge else "application/json")
    
    return model.generate_content(prompt).text

def try_get_answer(user_question, context="", fine_tuned_knowledge = False):

    parsed_result = {}
    if not fine_tuned_knowledge:
        response_json_valid = False
        is_expected_json = False
        max_attempts = 3
        while not response_json_valid and max_attempts > 0:
            response = ""

            try:
                response = generate_response(user_question, context , fine_tuned_knowledge)
            except Exception as e:
                print(f"Failed to create response for the question:\n{user_question}\n\n Error Code: {str(e)}")
                max_attempts = max_attempts - 1
                st.toast(f"Failed to create a response for your query.\n Error Code: {str(e)} \nTrying again... Retries left: {max_attempts} attempt/s")
                continue

            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid == False:
                print(f"Failed to validate and parse json for the questions:\n {user_question}")
                max_attempts = max_attempts - 1
                st.toast(f"Failed to validate and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue

            is_expected_json = is_expected_json_content(parsed_result)  
            if is_expected_json == False:
                print(f"Successfully validated and parse json for the question: {user_question} but is not on expected format... Trying again...")
                st.toast(f"Successfully validated and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue
            
            break
    else:
        try:
            print("Getting fine tuned knowledge...")
            parsed_result = generate_response(user_question, context , fine_tuned_knowledge)
        except Exception as e:
            print(f"Failed to create response for the question:\n\n {user_question}")
            parsed_result = "" 
            st.toast(f"Failed to create a response for your query.")

    return parsed_result

def user_input(user_question, api_key):
    
    with st.spinner("Processing..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        context = "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        parsed_result = try_get_answer(user_question, context)

        if "Is_Answer_In_Context" in parsed_result and not parsed_result["Is_Answer_In_Context"]:
            st.toast("Answer not found in the selected document. Attempting to generate an answer based on fine-tuned knowledge...")
            parsed_result = try_get_answer(user_question, context="", fine_tuned_knowledge=True)
    
    return parsed_result

def process_all_documents(api_key):
    retrievers_ref = st.session_state.db.collection('Retrievers')
    docs = retrievers_ref.stream()
    all_files = []
    for doc in docs:
        retriever = doc.to_dict()
        retriever['id'] = doc.id
        file_path, file_name = download_file_to_temp(retriever['document'])
        all_files.append(file_path)
    
    raw_text = get_pdf_text(all_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks, api_key)
    
def app():

    google_ai_api_key = st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]
    #Get firestore client
    firestore_db = firestore.client()
    st.session_state.db = firestore_db

    # Center the logo image
    col1, col2, col3 = st.columns([3, 4, 3])

    with col1:
        st.write(' ')

    with col2:
        st.image("Connext_Logo.png", width=250) 

    with col3:
        st.write(' ')

    st.markdown('## Welcome to :blue[Connext Chatbot] :robot_face:')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'parsed_result' not in st.session_state:
        st.session_state.parsed_result = {}

    chat_history_placeholder = st.empty()

    def display_chat_history():
        with chat_history_placeholder.container():
            st.markdown("""
                <style>
                .user-message {
                    background-color: #DCF8C6;
                    color: #000000;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 5px;
                    width: fit-content;
                    max-width: 70%;
                    word-wrap: break-word;
                    font-size: 16px;
                }
                .bot-message {
                    background-color: #F1F0F0;
                    color: #000000;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 5px;
                    width: fit-content;
                    max-width: 70%;
                    word-wrap: break-word;
                    font-size: 16px.
                }
                .user-message-container {
                    display: flex;
                    justify-content: flex-end.
                }
                .bot-message-container {
                    display: flex;
                    justify-content: flex-start.
                }
                </style>
            """, unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                st.markdown(f"""
                <div class="user-message-container">
                    <div class="user-message">{chat['question']}</div>
                </div>
                <div class="bot-message-container">
                    <div class="bot-message">{chat['answer']['Answer']}</div>
                </div>
                """, unsafe_allow_html=True)

    display_chat_history()

    # Include JavaScript to handle Enter key press
    st.markdown("""
        <script>
        const chatInput = window.parent.document.querySelector('textarea[aria-label="Ask a Question"]');
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                const submitButton = window.parent.document.querySelector('button[aria-label="Submit"]');
                submitButton.click();
            }
        });
        </script>
    """, unsafe_allow_html=True)

    user_question = st.chat_input("Ask a Question", key="user_question")

    if user_question:
        # Check for casual greetings
        if user_question.lower() in ["hey chatbot", "hello chatbot", "hi chatbot", "hello", "hi", "hey", "hello there", "hi there", "hey there"]:
            greeting_response = "Hello! How can I assist you today?"
            st.session_state.chat_history.append({"question": user_question, "answer": {"Answer": greeting_response}})
        else:
            if google_ai_api_key:
                parsed_result = user_input(user_question, google_ai_api_key)
                st.session_state.parsed_result = parsed_result
                if "Answer" in parsed_result:
                    st.session_state.chat_history.append({"question": user_question, "answer": parsed_result})
                    display_chat_history()
                else:
                    st.toast("Failed to get a valid response from the model.")

    display_chat_history()

    with st.sidebar:
        st.title("Processing Documents...")
        process_all_documents(google_ai_api_key)
        st.success("All documents have been processed and are ready for search.")

if __name__ == "__main__":
    app()
