import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from firebase_admin import credentials
from urllib.parse import urlparse, unquote
import os
import tempfile
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import datetime
import requests
import json

### Functions: Start ###

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

def load_creds():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'connext_chatbot_auth.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

creds = load_creds()

def download_file_to_temp(url):
    storage_client = storage.Client.from_service_account_info(st.session_state["connext_chatbot_admin_credentials"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
    temp_dir = tempfile.mkdtemp()

    response = requests.get(url)
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

    if not all(key in data for key in required_keys):
        return False
    
    return True

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
    genai.configure(credentials=creds)


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
                st.toast(f"Failed to create a response for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
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
        st.session_state.show_fine_tuned_expander = True
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        context = "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        parsed_result = try_get_answer(user_question, context)
        print(f"Parsed Result: {parsed_result}")
    
    return parsed_result

### App Function ###
def app():
    google_ai_api_key = st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]

    if not google_ai_api_key:
        st.error("Google API key is missing. Please provide it in the secrets configuration.")
        return

    firestore_db = firestore.client()
    st.session_state.db = firestore_db

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

    user_question = st.text_input("Ask a Question", key="user_question")
    submit_button = st.button("Submit", key="submit_button")

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
        st.session_state.show_fine_tuned_expander = True

    if 'parsed_result' not in st.session_state:
        st.session_state.parsed_result = {}

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

    if submit_button:
        if user_question and google_ai_api_key:
            st.session_state.parsed_result = user_input(user_question, google_ai_api_key)

    answer_placeholder = st.empty()

    if st.session_state.parsed_result is not None and "Answer" in st.session_state.parsed_result:
        answer_placeholder.write(f"Reply:\n\n {st.session_state.parsed_result['Answer']}")
        
        if "Is_Answer_In_Context" in st.session_state.parsed_result and not st.session_state.parsed_result["Is_Answer_In_Context"]:
            if st.session_state.show_fine_tuned_expander:
                with st.expander("Get fine-tuned answer?", expanded=False):
                    st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
                    col1, col2, _ = st.columns([3,3,6])
                    with col1:
                        if st.button("Yes", key="yes_button"):
                            st.session_state["request_fine_tuned_answer"] = True
                            st.session_state.show_fine_tuned_expander = False
                            st.rerun()
                    with col2:
                        if st.button("No", key="no_button"):
                            st.session_state.show_fine_tuned_expander = False
                            st.rerun()

    if st.session_state["request_fine_tuned_answer"]:
        fine_tuned_result = try_get_answer(user_question, context="", fine_tuned_knowledge=True)
        if fine_tuned_result:
            answer_placeholder.write(f"Fine-tuned Reply:\n\n {fine_tuned_result.strip()}")
            st.session_state.show_fine_tuned_expander = False
        else:
            answer_placeholder.write("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False

if __name__ == "__main__":
    app()
