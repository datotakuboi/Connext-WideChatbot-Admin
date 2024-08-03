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

@st.cache_resource
def google_oauth_link(flow):
    auth_url, _ = flow.authorization_url(redirect_uris=st.secrets["web"]["redirect_uris"], prompt='consent')
    st.write("Please go to this URL and authorize access:")
    st.markdown(f"[Sign in with Google]({auth_url})", unsafe_allow_html=True)
    code = st.text_input("Enter the authorization code:")
    return code

def load_creds():
    """Load credentials from Streamlit secrets and handle them using a temporary file."""
    # Parse the token data from Streamlit's secrets
    token_info = {
        'token': st.secrets["token"]["value"],
        'refresh_token': st.secrets["token"]["refresh_token"],
        'token_uri': st.secrets["token"]["token_uri"],
        'client_id': st.secrets["token"]["client_id"],
        'client_secret': st.secrets["token"]["client_secret"],
        'scopes': st.secrets["token"]["scopes"],
        'expiry': st.secrets["token"]["expiry"]  # Assuming expiry is directly usable
    }

    # Create a temporary file to store the token data
    temp_dir = tempfile.mkdtemp()
    token_file_path = os.path.join(temp_dir, 'token.json')
    with open(token_file_path, 'w') as token_file:
        json.dump(token_info, token_file)

    # Load the credentials from the temporary file
    creds = Credentials.from_authorized_user_file(token_file_path, SCOPES)

    # Refresh the token if necessary
    if creds and creds.expired and creds.refresh_token:
        st.toast("Currently refreshing token...")
        creds.refresh(Request())

        # Optionally update the temporary file with the refreshed token data
        with open(token_file_path, 'w') as token_file:
            token_file.write(creds.to_json())

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

    # Save the content to the file
    blob.download_to_filename(temp_file_path)

    return temp_file_path, file_name

def extract_and_parse_json(text):
    # Find the first opening and the last closing curly brackets
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if (start_index == -1 or end_index == -1 or end_index < start_index):
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

            #Test 1
            try:
                response = generate_response(user_question, context , fine_tuned_knowledge)
            except Exception as e:
                print(f"Failed to create response for the question:\n{user_question}\n\n Error Code: {str(e)}")
                max_attempts = max_attempts - 1
                st.toast(f"Failed to create a response for your query.\n Error Code: {str(e)} \nTrying again... Retries left: {max_attempts} attempt/s")
                continue

            #Test 2
            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid == False:
                print(f"Failed to validate and parse json for the questions:\n {user_question}")
                max_attempts = max_attempts - 1
                st.toast(f"Failed to validate and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue

            #Test 3
            is_expected_json = is_expected_json_content(parsed_result)  
            if is_expected_json == False:
                print(f"Successfully validated and parse json for the question: {user_question} but is not on expected format... Trying again...")
                st.toast(f"Successfully validated and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue
            
            break #If all tests passed above
    else: #if using fine_tuned knowledge
        try:
            print("Getting fine tuned knowledge...")
            parsed_result = generate_response(user_question, context , fine_tuned_knowledge)
        except Exception as e:
            print(f"Failed to create response for the question:\n\n {user_question}")
            parsed_result = "" #Defaul empty string given when failed to generate response
            st.toast(f"Failed to create a response for your query.")

    return parsed_result

def user_input(user_question, api_key, chat_history):
    with st.spinner("Processing..."):
        st.session_state.show_fine_tuned_expander = True  # Reset
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Create context from chat history
        context = "\n\n--------------------------\n\n".join([f"User: {entry['question']}\nBot: {entry['answer']['Answer']}" for entry in chat_history])
        context += "\n\n--------------------------\n\n"
        context += "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        parsed_result = try_get_answer(user_question, context)
        print(f"Parsed Result: {parsed_result}")
    
    return parsed_result


def app():
    google_ai_api_key = st.secrets["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]
    #Get firestore client
    if not firebase_admin._apps:
        firestore_db = firebase_admin.initialize_app(credentials.Certificate(st.session_state["connext_chatbot_admin_credentials"]))
    else:
        firestore_db = firebase_admin.get_app()

    st.session_state.db = firestore.client(firestore_db)

    # Center the logo image
    col1, col2, col3 = st.columns([3,4,3])

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

    # Display chat history above the input section
    chat_history_placeholder = st.empty()
    answer_placeholder = st.empty()  # Placeholder for answers

    def display_chat_history():
        with chat_history_placeholder.container():
            for chat in st.session_state.chat_history:
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**Bot:** {chat['answer']['Answer']}")

    display_chat_history()

    user_question = st.text_input("Ask a Question", key="user_question")
    submit_button = st.button("Submit", key="submit_button")
    clear_history_button = st.button("Clear Chat History")

    if clear_history_button:
        st.session_state.chat_history = []
        display_chat_history()  # Refresh the chat history display

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

    if submit_button:
        if user_question and google_ai_api_key:
            parsed_result = user_input(user_question, google_ai_api_key, st.session_state.chat_history)
            st.session_state.parsed_result = parsed_result
            st.session_state.chat_history.append({"question": user_question, "answer": parsed_result})
            display_chat_history()  # Update chat history display

            # Display the answer and the fine-tuning option if applicable
            if st.session_state.parsed_result is not None and "Answer" in st.session_state.parsed_result:
                answer_placeholder.write(f"Reply:\n\n {st.session_state.parsed_result['Answer']}")
                
                # Check if the answer is not directly in the context
                if "Is_Answer_In_Context" in st.session_state.parsed_result and not st.session_state.parsed_result["Is_Answer_In_Context"]:
                    if st.session_state.show_fine_tuned_expander:
                        with st.expander("Get fine-tuned answer?", expanded=False):
                            st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
                            col1, col2, _ = st.columns([3,3,6])
                            with col1:
                                if st.button("Yes", key="yes_button"):
                                    # Use session state to handle the rerun after button press
                                    st.session_state["request_fine_tuned_answer"] = True
                                    st.session_state.show_fine_tuned_expander = False
                                    st.rerun()
                            with col2:
                                if st.button("No", key="no_button"):
                                    st.session_state.show_fine_tuned_expander = False
                                    st.rerun()

    # Handle the generation of fine-tuned answer if the flag is set
    if st.session_state["request_fine_tuned_answer"]:
        fine_tuned_result = try_get_answer(user_question, context="", fine_tuned_knowledge=True)
        if fine_tuned_result:
            answer_placeholder.write(f"Fine-tuned Reply:\n\n {fine_tuned_result.strip()}")

            # Update chat history with fine-tuned answer
            st.session_state.chat_history[-1]['answer'] = {"Answer": fine_tuned_result.strip()}
            st.session_state.show_fine_tuned_expander = False
        else:
            answer_placeholder.write("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False  # Reset the flag after handling

    with st.sidebar:
        st.title("PDF Documents:")
        for idx, doc in enumerate(docs, start=1):
            retriever = doc.to_dict()
            retriever['id'] = doc.id  # Add document ID to the retriever dictionary
            retriever_name = retriever['retriever_name']
            retriever_description = retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")
                file_path, file_name = download_file_to_temp(retriever['document']) # Get the document file path and file name
                st.markdown(f"_**File Name**_: {file_name}")
                retriever["file_path"] = file_path 
                st.session_state["retrievers"][retriever_name] = retriever #populate the retriever dictionary
        st.title("PDF Document Selection:")
        st.session_state["selected_retrievers"] = st.multiselect("Select Documents", list(st.session_state["retrievers"].keys()))  
        
        #Get pdf docs of selected retrievers from st.session_state["selected_retrievers"]
        if st.button("Submit & Process", key="process_button"):
            if google_ai_api_key:
                with st.spinner("Processing..."):
                    # Get pdf docs of selected retrievers from st.session_state["selected_retrievers"]
                    selected_files = [st.session_state["retrievers"][name]["file_path"] for name in st.session_state["selected_retrievers"]]
                    raw_text = get_pdf_text(selected_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, google_ai_api_key)
                    st.success("Done")
            else:
                st.toast("Failed to process the documents", icon="💥")

if __name__ == "__main__":
    app()
