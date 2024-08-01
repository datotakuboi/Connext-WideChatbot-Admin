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
import mimetypes
#Google Auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
#external python files
import configuration, playground


if "api_keys" not in st.session_state:
    st.session_state["api_keys"] = {}
    

if "connext_chatbot_admin_credentials" not in st.session_state:
    st.session_state["connext_chatbot_admin_credentials"] = None

if "is_streamlit_deployed" not in st.session_state:
    st.session_state["is_streamlit_deployed"] = True

if "oauth_creds" not in st.session_state:
    st.session_state["oauth_creds"] = None

#Configure this one to True if deployed on streamlit community cloud or on local machine
#This helps change the json file and api key loading
st.session_state["is_streamlit_deployed"] = True
firebase_api_key = None
google_ai_api_key = None

def get_credentials_dict(credentials_input):
    if isinstance(credentials_input, str):
        try:
            # Attempt to parse the string as JSON
            return json.loads(credentials_input)
        except json.JSONDecodeError:
            raise ValueError("Credentials string is not valid JSON")
    elif isinstance(credentials_input, dict):
        return credentials_input
    else:
        raise ValueError("Credentials must be a JSON string or a dictionary")

if st.session_state["is_streamlit_deployed"]:
    # Load the JSON content from Streamlit secrets
    service_account_info = st.secrets["gcp_service_account"]
    # Convert the TOML object to a dictionary
    st.session_state["connext_chatbot_admin_credentials"] = dict(service_account_info)
    st.session_state["api_keys"]["FIREBASE_API_KEY"] = st.secrets["FIREBASE_API_KEY"]
    st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"] = st.secrets["GOOGLE_AI_STUDIO_API_KEY"]
    firebase_api_key = st.secrets["FIREBASE_API_KEY"]
    google_ai_api_key = st.secrets["GOOGLE_AI_STUDIO_API_KEY"]
else:
    with open('connext-chatbot-admin-ce0eb842ce8e.json') as f:
        st.session_state["connext_chatbot_admin_credentials"] = json.load(f)
    load_dotenv(dotenv_path='cred.env')  # This method will read key-value pairs from a .env file and add them to environment variable.
    firebase_api_key = os.getenv('FIREBASE_API_KEY')
    google_ai_api_key = os.getenv('GOOGLE_AI_STUDIO_API_KEY')
    st.session_state["api_keys"]["FIREBASE_API_KEY"] = firebase_api_key
    st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"] = google_ai_api_key 

#Firebase SDK initialization
if not firebase_admin._apps:
    cred = credentials.Certificate(st.session_state["connext_chatbot_admin_credentials"])
    firebase_admin.initialize_app(cred)

### For Log-in Page: Start ###

## Functions: Start ###
@st.experimental_dialog("Logging In Failed")
def fail_login_dialog(message):
    st.markdown(message)
    st.markdown("Please try again or contact administrator")

def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        
        try:
            payload = {
                "returnSecureToken": return_secure_token
            }
            if email:
                payload["email"] = email
            if password:
                payload["password"] = password
            payload = json.dumps(payload)
            #print('payload sigin',payload)
            r = requests.post(rest_api_url, params={"key": firebase_api_key}, data=payload)
            data = r.json()
            try:
                user_info = {
                    'email': data['email'],
                    'username': data.get('displayName')  # Retrieve username if available
                }
                st.toast("Logged In Succesfully", icon="🎉")
                return user_info, True #User Info and Login Status
            except:
                fail_login_dialog("Failed to login: " + data.get("error", {}).get("message", "No error message provided"))
                return None, False #User Info and Login Status
        except Exception as e:
            fail_login_dialog(f"Error during login: {str(e)}")
            return None, False #User Info and Login Status

def reset_password(email):
    try:
        rest_api_url = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
        payload = {"email": email, "requestType": "PASSWORD_RESET"}
        payload = json.dumps(payload)
        r = requests.post(
            rest_api_url,
            params={"key": firebase_api_key},
            data=payload,
        )
        if r.status_code == 200:
            return True, "Reset email Sent"
        else:
            # Handle error response
            error_message = r.json().get("error", {}).get("message")
            return False, error_message
    except Exception as e:
        return False, str(e)

@st.experimental_dialog("Forgot Password")
def forget():
    email = st.text_input('Email')
    if st.button('Send Reset Link'):
        print(email)
        success, message = reset_password(email)
        if success:
            st.success("Password reset email sent successfully.")
        else:
            st.warning(f"Password reset failed: {message}") 

def login():
    userinfo, login_status = sign_in_with_email_and_password(st.session_state.email_input,st.session_state.password_input)
    if userinfo:
        st.session_state.username = userinfo['username']
        st.session_state.useremail = userinfo['email']

    if login_status:
        st.session_state.signedout = False
    else:
        st.session_state.signedout = True

## Functions: End ###

if 'db' not in st.session_state:
        st.session_state.db = ''

if 'username' not in st.session_state:
    st.session_state.username = ''

if 'useremail' not in st.session_state:
    st.session_state.useremail = ''

if "signedout"  not in st.session_state:
    st.session_state["signedout"] = True #By default the user is signed out during intial loading

if st.session_state["signedout"]: #If user is on a state of signed out
    st.markdown('## :blue[Connext Chatbot Admin] Log In :robot_face:')
    email = st.text_input('Email Address', key="email_add")
    password = st.text_input('Password',type='password', key="password")
    st.session_state.email_input = email
    st.session_state.password_input = password

    st.button('Login', on_click=login)
    st.button("Forgot Password", on_click=forget)

if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = {}

if not st.session_state["signedout"]: #If user is on a state of signed out

    with st.sidebar:
        selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Configuration", "Playground"],  # required
        icons=["wrench", "robot"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        )

    if selected == "Configuration":
        configuration.app()
    elif selected == "Playground":
        playground.playground()  # Corrected this line to call the function directly

### For Log-in Page: End ###
