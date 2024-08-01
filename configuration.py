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



### Functions: Start ###

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

    # # Save the content to the file
    # with open(temp_file_path, 'wb') as temp_file:
    #     temp_file.write(response.content)
    blob.download_to_filename(temp_file_path)

    return temp_file_path, file_name

def update_file(file, retriever):
    
    @st.experimental_dialog("Update Failed")
    def fail_update_dialog(message):
        st.markdown(message)
        st.markdown("Please ensure that a document was uploaded before updating")

    if file is not None:
        print(f"Status: Uploading {file.name} for retriever: {retriever['retriever_name']}")
        # Initialize Firebase Storage client
        storage_client = storage.Client.from_service_account_json('connext-chatbot-admin-ce0eb842ce8e.json')
        bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

        # Delete the old document from Firebase Storage
        # Parse the old file URL to get the file name
        old_file_url = retriever['document']
        print(f"Old file URL: {old_file_url}")
        old_file_path = urlparse(old_file_url).path
        old_file_name = os.path.basename(unquote(old_file_path))
        print(f"Parsed old file name: {old_file_name}")
        old_blob = bucket.blob(old_file_name)

        if old_blob.exists():
            print(f"Deleting old file: {old_file_path}")
            old_blob.delete()
        else:
            print(f"Old file not found: {old_file_path}")

        # Upload the new document to Firebase Storage
        file_path = file.name
        new_blob = bucket.blob(file_path)
        
        # Set MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/pdf'  # Default to PDF if MIME type cannot be guessed
        
        new_blob.upload_from_file(file, content_type=mime_type)
        download_url = new_blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        
        # Update the retriever dictionary
        retriever['document'] = download_url
        st.session_state.db.collection('Retrievers').document(retriever['id']).update({'document': download_url})
        st.session_state["retrievers"][retriever['retriever_name']]['file_name'] = file_path
        st.session_state["retrievers"][retriever['retriever_name']]['document'] = download_url

        print(f"File {file_path} uploaded successfully with MIME type {mime_type} and retriever document updated.")

    else:
        fail_update_dialog(f"No file was selected to update for {retriever['retriever_name']}")

    return None

@st.experimental_dialog("Document Deletion Confirmation")
def delete_retriever(retriever):
    retriever_name = retriever['retriever_name']
    st.markdown(f"Are you sure you want to delete the following document: \"{retriever_name}\"?")
    if st.button(f"Confirm Delete {retriever_name}", key=f"confirm_delete_{retriever_name}"):
        # Initialize Firebase Storage client
        storage_client = storage.Client.from_service_account_info(st.session_state["connext_chatbot_admin_credentials"])
        bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

        doc_id = retriever['id']
        document_url = retriever['document']

        # Parse the document URL to get the file name
        file_path = urlparse(document_url).path
        file_name = os.path.basename(unquote(file_path))
        blob = bucket.blob(file_name)

        # Delete the file from Firebase Storage
        if blob.exists():
            blob.delete()

        # Delete the retriever document from Firestore
        st.session_state.db.collection('Retrievers').document(doc_id).delete()

        # Delete the retriever from local session state
        del st.session_state["retrievers"][retriever_name]

        st.toast(f"Document {retriever_name} Deleted Successfully", icon="🗑️")
        st.rerun()  # Refresh the page to update the retriever list

@st.experimental_dialog("New Document")
def add_retriever():
    name = st.text_input("Document Name", key="new_retriever")
    description = st.text_area("Document Description", height=300)
    file = st.file_uploader("Upload Chatbot Document", accept_multiple_files=False, type=["pdf", "doc", "docx"])

    if st.button("Submit"):
        # Check if any of the required fields are empty and show appropriate warning messages
        if not name:
            st.warning("Please enter the Document Name")
        if not description:
            st.warning("Please enter the Document Description")
        if file is None:
            st.warning("Please upload a Chatbot Document")

        if name and description and file:
            # Initialize Firebase Storage client
            storage_client = storage.Client.from_service_account_json('connext-chatbot-admin-3d098c02afad.json')
            bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

            # Upload the document to Firebase Storage
            file_path = file.name
            new_blob = bucket.blob(file_path)

            # Set MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = 'application/pdf'  # Default to PDF if MIME type cannot be guessed

            new_blob.upload_from_file(file, content_type=mime_type)
            download_url = new_blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

            # Add the new retriever document to Firestore
            retriever_data = {
                'retriever_name': name,
                'retriever_description': description,
                'document': download_url
            }
            doc_ref = st.session_state.db.collection('Retrievers').add(retriever_data)

            # Update the local session state
            retriever_data['id'] = doc_ref[1].id  # Add the document ID
            st.session_state["retrievers"][name] = retriever_data
            st.toast("New Document Added Successfully", icon="🎉")
            st.rerun()  # Refresh the page to show the new retriever

@st.experimental_dialog("Update Document Description")
def update_description(retriever):
    #Get the retriever dictionary and retriever description then update firebase documents then update the local retriever information
    #Update the local memory st.session_state.db.collection('Retrievers') contents
    
    def update_action(new_description):
        try:
            print("Updating description...")
            st.session_state.db.collection('Retrievers').document(retriever['id']).update({'retriever_description': new_description})
            st.session_state["retrievers"][retriever['retriever_name']]['retriever_description'] = new_description
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        st.toast("Document Description Updated Succesfully", icon="🎉")

    st.markdown(retriever["retriever_name"])
    description = st.text_area("Document Description", height=300, value=retriever["retriever_description"], key=f"description_{retriever['retriever_name']}")
    if st.button("Update",  key=f"update_desc_button_{retriever['retriever_name']}"):
        update_action(description)

@st.experimental_dialog("Update Document Name")
def update_name(retriever):
    #Get the retriever dictionary and retriever name, then update firebase documents then update the local retriever information
    #Update the local memory st.session_state.db.collection('Retrievers') contents

    def update_action(new_name):
        try:
            st.session_state.db.collection('Retrievers').document(retriever['id']).update({'retriever_name': new_name})
            st.session_state["retrievers"][new_name] = st.session_state["retrievers"].pop(retriever['retriever_name'])
            st.session_state["retrievers"][new_name]['retriever_name'] = new_name
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        st.toast("Document Name Updated Succesfully", icon="🎉")

    st.markdown(retriever["retriever_name"])
    new_name = st.text_input("Retriever Name", value=retriever["retriever_name"], key=f"name_{retriever['retriever_name']}")
    if st.button("Update", key=f"update_name_button_{retriever['retriever_name']}"):
        update_action(new_name)

### Functions: End ###


def app():
    #Get firestore client
    firestore_db=firestore.client()
    st.session_state.db=firestore_db

    # Center the logo image
    col1, col2, col3 = st.columns([3,4,3])

    with col1:
        st.write(' ')

    with col2:
        st.image("Connext_Logo.png", width=250) 

    with col3:
        st.write(' ')

    st.markdown('## Welcome to :blue[Connext Chatbot Admin] :robot_face:')
    st.markdown('---')
    st.markdown("### Connext Chatbot Documents: ")

    retrievers_ref = st.session_state.db.collection('Retrievers')
    docs = retrievers_ref.stream()

    
    # Display the retrievers
    for idx, doc in enumerate(docs, start=1):
        retriever = doc.to_dict()
        retriever['id'] = doc.id  # Add document ID to the retriever dictionary
        #print(retriever)
        retriever_name = retriever['retriever_name']
        retriever_description = retriever['retriever_description']
        st.markdown(f"##### {idx}. {retriever_name}")

        try:
            file_path, file_name = download_file_to_temp(retriever['document']) # Get the document file path and file name
        except:
            st.warning(f"Unable to load the following pdf: {retriever['retriever_name']}. Please contact administrator")
            continue
        
        with st.expander("Document Information"):
            col1, col2, col3, col4 = st.columns([4,4,3,3])
            with col1:
                if st.button("Edit Document Name", key=f"{retriever_name}_retriever_name_editor"):
                    update_name(retriever)
            with col2:
                if st.button("Delete Document", key=f"{retriever_name}_retriever_delete"):
                    delete_retriever(retriever)
            st.markdown(f"**Description:** {retriever_description}")
            if st.button("Edit Description", key=f"{retriever_name}_retriever_description_editor"):
                 update_description(retriever)
            retriever_dict = {
                "retriever_description":retriever_description,
                "file_name":file_name
            }
            st.session_state["retrievers"][retriever_name] = retriever_dict
            st.markdown(f"_**File Name**_: {file_name}")
            updated_doc = st.file_uploader("Upload Chatbot Document", accept_multiple_files=False, type=["pdf", "doc", "docx"], key=f"{retriever_name}_file_uploader")
            st.button("Update File", on_click=partial(update_file, updated_doc, retriever), key=f"{retriever_name}_file_update_button")
    
    st.markdown('---')
    if st.button("Add New Document"):
        add_retriever()