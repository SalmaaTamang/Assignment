
import streamlit as st
import requests
import json
import time
from uuid import uuid4
import pandas as pd

# Base URL for the FastAPI backend
BASE_URL = "http://127.0.0.1:8000"

API_URL = "http://localhost:8000/callback"
# Define the API endpoints
CHAT_HISTORY_URL = "http://localhost:8000/chathistory/"  

def call_api(endpoint, method="GET", data=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data)  # Send data as form-encoded
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return {}

# Page 1: Landing Page
def landing_page():
   
    st.title("Welcome to the 🏠 Landing Page of 💬 Chat with 📄 Docs")
    

# Page 2: Create Ingestion
def create_ingestion():
    st.title(" 📥 Create Ingestion")
    kb_name = st.text_input("Enter Knowledge Base Name:")
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=["pdf", "docx", "txt", "jpg", "jpeg", "png"])
    
    if st.button("Submit"):
        if not kb_name or not uploaded_files:
            st.error("Please provide a Knowledge Base Name and upload files.")
        else:
            # Prepare files for upload
            files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
            data = {"kb_name": kb_name}
            response = requests.post(f"{BASE_URL}/upload/", data=data, files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                task_id = result["task_id"]
                st.write(f"Task ID: {task_id}")
                
                # Track background task status
                with st.spinner("Tracking Task Status..."):
                    while True:
                        task_status = call_api(f"/task_status/{task_id}")
                        if task_status["status"] == "success":
                            st.success("Ingestion completed successfully!")
                            break
                        elif task_status["status"] == "failed":
                            st.error("Ingestion failed.")
                            break
                        time.sleep(2)  # Poll every 2 seconds
            else:
                st.error("Failed to upload files.")

# page3 : chat and chat history

def fetch_chat_history():
    response = requests.get("http://localhost:8000/chathistory/")
    if response.status_code == 200:
        chat_history = response.json()
        return {item["title"]: item["conv_id"] for item in chat_history}
    else:
        st.sidebar.write("Failed to load chat history.")
        return {}

# Display chat history in the sidebar
def display_chat_history(chat_titles):
    for title, conv_id in chat_titles.items():
        if st.sidebar.button(f"**{title}**", key=f"title_{conv_id}"):
            st.session_state.selected_chat = conv_id

# Display chat content for the selected conversation
def display_chat_content(conv_id):
    response = requests.get(f"http://localhost:8000/chathistory/{conv_id}")
    if response.status_code == 200:
        chat_content = response.json()
        for entry in chat_content:
            st.subheader("❓")
            st.write(entry["query"])
            st.subheader("💡")
            st.write(entry["response"])
    else:
      st.write("Failed to load chat content.")
def chat_page():
    st.title("💬 Chat with us. ")
    
    # Sidebar: Knowledge Base Selection (Dropdown)
    st.sidebar.header("Select Knowledge Base")
    kb_metadata = call_api("/kb_metadata/")
    if isinstance(kb_metadata, dict) and "error" in kb_metadata:
        st.sidebar.error(f"Error fetching knowledge bases: {kb_metadata['error']}")
    elif not isinstance(kb_metadata, list):
        st.sidebar.error("Unexpected response format from server. Please try again.")
    else:
        kb_names = [kb["kb_name"] for kb in kb_metadata]
        selected_kb = st.sidebar.selectbox("Choose a Knowledge Base:", kb_names)

    # Sidebar: Chat History
    st.sidebar.header("Chat History")
    if "chat_titles" not in st.session_state:
        st.session_state.chat_titles = {}
    if "selected_chat" not in st.session_state:
        st.session_state.selected_chat = None
    
    # Fetch and display chat history
    st.session_state.chat_titles = fetch_chat_history()
    display_chat_history(st.session_state.chat_titles)
    
    # Display selected chat content
    if st.session_state.selected_chat:
        display_chat_content(st.session_state.selected_chat)
    
    # Start a new chat session
    st.sidebar.markdown("---")  # Add a separator for better UI
    if st.sidebar.button("New Chat +"):
        if 'selected_kb' in locals():  # Ensure a knowledge base is selected
            response = call_api("/newchat/", method="POST", data={"kb_name": selected_kb})
            conv_id = response["conv_id"]
            st.session_state.conv_id = conv_id
            st.success(f"New Chat Created! Conversation ID: {conv_id}")
        else:
            st.sidebar.warning("Please select a knowledge base first.")
    
    # Ensure a conversation ID exists
    if "conv_id" not in st.session_state:
        st.warning("Please start a new chat session.")
        return
    
    conv_id = st.session_state.conv_id
    
    # Display chat history
    chat_history = call_api(f"/chathistory/{conv_id}")
    for entry in chat_history:
        st.markdown(f"**User:** {entry['query']}")
        st.markdown(f"**Assistant:** {entry['response']}")
        
        # Display chunk texts in a collapsible expander
        if "chunk_texts" in entry and isinstance(entry["chunk_texts"], list):
            with st.expander("Source", expanded=False):  # Collapsible section
                st.markdown("**Sources:**")
                for i, chunk_text in enumerate(entry["chunk_texts"]):
                    st.markdown(f"- **source {i+1}:** {chunk_text}")
    
    # Input field for user query
    user_input = st.chat_input("Enter your query:")
    
    # Handle user input directly from chat_input
    if user_input:
        response = call_api("/chat/", method="POST", data={
            "conv_id": conv_id,
            "user_input": user_input
        })
        if "response" in response:
            st.success(f"Response: {response['response']}")
            
            # Handle chunk_texts returned by /chat/
            if "chunk_texts" in response and isinstance(response["chunk_texts"], list):
                with st.expander("Source", expanded=False):  # Collapsible section
                    st.markdown("**Sources:**")
                    for i, chunk_text in enumerate(response["chunk_texts"]):
                        st.markdown(f"- **source {i+1}:** {chunk_text}")
            else:
                st.warning("No chunk texts found in the response.")
        else:
            st.error("Unexpected response from server. Please try again.")

def book_appointment():

    st.title("📅 Book an Appointment")
    
    # Initialize session state variables
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # Ensure session ID is stored persistently
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Store conversation history
    if "show_chat_input" not in st.session_state:
        st.session_state.show_chat_input = False  # Control visibility of chat input

    def start_conversation():
        response = requests.post(API_URL, json={"session_id": None})  # No session ID initially
        if response.status_code == 200:
            data = response.json()

            if "session_id" in data:
                st.session_state.session_id = data["session_id"]  # Store session ID persistently
                st.write(f"DEBUG: Session ID stored in state: {st.session_state.session_id}")
                return data["response"]  # Return the bot's first message
            else:
                st.error("API did not return a session ID.")
        else:
            st.error("Failed to start a new conversation. Please try again.")
            st.write("DEBUG: API Response Error:", response.text)
        
        return None

    def send_message(user_input):
        """Send a user message to the backend and get the bot's response."""
        if not st.session_state.session_id:
            st.error("No active session. Please start a new conversation.")
            return None

        # Use the session_id stored in st.session_state
        session_id_to_send = st.session_state.session_id
        print("DEBUG: Sending session_id to backend:", session_id_to_send)  # Debugging

        # Log the full payload being sent to the backend
        payload = {
            "session_id": session_id_to_send,
            "user_input": user_input
        }
        print("DEBUG: Payload sent to backend:", payload)  # Debugging

        # Send the request to the backend
        response = requests.post(API_URL, json=payload)

        # Print the full response JSON for debugging
        try:
            response_json = response.json()  # Extract JSON data
            print("DEBUG: Full Response JSON from Backend:", response_json)  # Debugging
        except Exception as e:
            st.error(f"Failed to parse response JSON: {e}")
            st.write("DEBUG: Raw Response Text:", response.text)
            return None

        # Check if the backend returned a new session_id (optional behavior)
        backend_session_id = response_json.get("session_id", None)
        if backend_session_id and backend_session_id != st.session_state.session_id:
            print("DEBUG: Backend returned a new session_id:", backend_session_id)
            print("DEBUG: Updating session_id in state to match backend.")
            st.session_state.session_id = backend_session_id  # Update session_id if necessary

        # Handle the response based on status code
        if response.status_code == 200:
            bot_response = response_json.get("response", "No response key in JSON")
            print("DEBUG: Bot Response:", bot_response)  # Debugging
            return bot_response
        else:
            st.error("Failed to send message. Please try again.")
            st.write("DEBUG: API Response Error:", response.text)
            return None


    # Debugging: Show current session ID
    st.write("DEBUG: Current Session ID:", st.session_state.session_id)

    # Button to start a new conversation
    if st.button("Book Appointment"):
        if not st.session_state.session_id:  # Only start a new session if none exists
            bot_response = start_conversation()
            if bot_response:
                st.session_state.conversation.append(("Bot", bot_response))  # Add bot's response to history
                st.session_state.show_chat_input = True  # Show chat input field
        else:
            st.warning("You already have an active session. Continue the conversation.")

    # Display the conversation history
    st.subheader("Conversation History")
    for role, text in st.session_state.conversation:
        st.write(f"**{role}:** {text}")

    # Show chat input only if the session has started
    if st.session_state.show_chat_input:
        user_input = st.chat_input("Type your message here...")
        if user_input and user_input.strip():  # Check if user input is not empty
            bot_response = send_message(user_input)
            if bot_response:
                st.session_state.conversation.append(("You", user_input))  # Add user's message to history
                st.session_state.conversation.append(("Bot", bot_response))  # Add bot's response to history
                st.rerun()  # Rerun to update the conversation history

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Go to",
        ["🏠 Landing Page", "📥 Create Ingestion", "💬 Chat", "📅 Book Appointment"]
    )

    if page == "🏠 Landing Page":
        landing_page()
    elif page == "📥 Create Ingestion":
        create_ingestion()
    elif page == "💬 Chat":
        chat_page()
    elif page == "📅 Book Appointment":
        book_appointment()

# Ensure this block runs only when the script is executed directly
if __name__ == "__main__":
    main()
