from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import lancedb
from lancedb.pydantic import LanceModel, Vector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from typing import List
from google.generativeai import GenerativeModel, configure  # Gemini API
from sentence_transformers import SentenceTransformer  # For embeddings
import numpy as np
from uuid import UUID, uuid4
from pydantic import Field, BaseModel
from fastapi import BackgroundTasks, Query
from datetime import datetime, timedelta
import dateparser
import re
import uuid
import logging
from fastapi import Body
import os
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

db_directory = "data/knowledgebase"
if not os.path.exists(db_directory):
    os.makedirs(db_directory)
db = lancedb.connect(db_directory)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

configure(api_key=gemini_api_key)
gemini_model = GenerativeModel('gemini-2.0-flash-exp')


# Define Pydantic models for tables
class TaskStatus(LanceModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    status: str
    class Config:
        primary_key = "task_id"

class KBMetadata(LanceModel):
    kb_id: str = Field(default_factory=lambda: str(uuid4()))
    kb_name: str
    uploaded_file_url: str
    class Config:
        primary_key = 'kb_id'

class KBEmbeddings(LanceModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    kb_id: str
    text: str
    vector: Vector(384)  # Adjusted for Hugging Face embeddings
    class Config:
        primary_key = 'chunk_id'
        foreign_keys = {'kb_id': 'KBMetadata.kb_id'}

class ChatLookUp(LanceModel):
    conv_id: str = Field(default_factory=lambda: str(uuid4()))
    kb_name: str
    class Config:
        primary_key = 'conv_id'

class ChatTable(LanceModel):
    ct_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    response: str
    chunk_ids: List[str]
    chunk_texts: List[str]
    similarity_scores: list[float]
    class Config:
        primary_key = "ct_id"
        foreign_keys = {'chunk_id': 'KBEmbeddings.chunk_id'}

class ChatHistory(LanceModel):
    ch_id: str = Field(default_factory=lambda: str(uuid4()))
    conv_id: str
    title: str
    class Config:
        primary_key = 'ch_id'
        foreign_keys = {'conv_id': 'ChatLookUp.conv_id'}

class ChatUsage(LanceModel):
    chu_id: str = Field(default_factory=lambda: str(uuid4()))
    ct_id: str
    input_token: int
    output_token: int
    total_token: int
    class Config:
        primary_key = "chu_id"
        foreign_keys = {'ct_id': "ChatTable.ct_id"}

class ConversationalFormData(LanceModel):
    form_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    email: str
    phone_number: str
    appointment_date: str
    class Config:
        primary_key = "form_id"

# Helper functions for text extraction
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_image(file_path: str) -> str:
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text(file_path: str, ext: str) -> str:
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return extract_text_from_image(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Generate embeddings and calculate cosine similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_model.encode(texts).tolist()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def generate_response(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# Helper function to get the next weekday date
def get_next_weekday(day_name):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    target_day = days.index(day_name.capitalize())
    today = datetime.now().date()
    current_day = today.weekday()
    days_to_add = (target_day - current_day + 7) % 7 or 7
    return today + timedelta(days=days_to_add)


# Parse natural language dates
def parse_natural_date(user_input):
    user_input = user_input.lower().strip()
    today = datetime.today().date()

    # Reject explicitly past dates
    if user_input in ["yesterday", "previous day", "last day"]:
        return None  # Invalid case

    # Handle relative dates
    if user_input == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif user_input in ["the day after tomorrow", "day after tomorrow"]:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    # Detect "X days ago" (invalid case)
    match_past = re.search(r'(\d+)\s*days ago', user_input)
    if match_past:
        return None  # Reject past dates

    # Detect "in X days"
    match_future = re.search(r'in (\d+)\s*days', user_input)
    if match_future:
        return (today + timedelta(days=int(match_future.group(1)))).strftime("%Y-%m-%d")

    # Handle weekdays
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days_of_week:
        if day in user_input:
            parsed_date = get_next_weekday(day.capitalize())
            if parsed_date < today:  # Ensure future dates
                return None
            return parsed_date.strftime("%Y-%m-%d")

    # Use dateparser to handle other cases
    parsed_date = dateparser.parse(user_input, settings={'PREFER_DATES_FROM': 'future'})
    if parsed_date:
        parsed_date = parsed_date.date()
        if parsed_date < today:
            return None  # Reject past dates
        return parsed_date.strftime("%Y-%m-%d")

    return None  # If parsing fails


# Centralized table management function
def get_or_create_table(table_name: str, schema, mode="create"):
    try:
        if table_name not in db.table_names():
            return db.create_table(table_name, schema=schema, mode=mode)
        else:
            return db.open_table(table_name)
    except ValueError as e:
        if "already exists" in str(e):
            return db.open_table(table_name)  # Open the existing table
        else:
            raise  # Re-raise the exception if it's unrelated to table existence

# Session Management
sessions = {}
def generate_session_id():
    return str(uuid.uuid4())

class ConversationalForm:
    def __init__(self):
        self.state = "start"
        self.data = {}

    def process_input(self, user_input, session_id):
        if self.state == "start":
            self.state = "name"
            return generate_response("Generate a friendly greeting to ask for the user's name.")
        elif self.state == "name":
            self.data["name"] = user_input
            self.state = "email"
            return generate_response(f"Generate a polite question asking for the email address of {user_input}.")
        elif self.state == "email":
            if not self.validate_email(user_input):
                return "Invalid email format. Please provide a valid email address."
            self.data["email"] = user_input
            self.state = "phone_number"
            return generate_response(f"Generate a polite question asking for the phone number of {self.data['name']}.")
        elif self.state == "phone_number":
            if not self.validate_phone_number(user_input):
                return "Invalid phone number format. Please provide a valid phone number."
            self.data["phone_number"] = user_input
            self.state = "appointment_date"
            return generate_response(f"Generate a polite question asking for the preferred appointment date of {self.data['name']}.")
        elif self.state == "appointment_date":
            appointment_date = parse_natural_date(user_input)
            if not appointment_date or datetime.strptime(appointment_date, "%Y-%m-%d").date() < datetime.today().date():
                return "Invalid date. Please provide a valid future date."
            self.data["appointment_date"] = appointment_date
            self.state = "complete"
            # Store data in LanceDB
            self.store_in_database(session_id)
            return (
                f"Thank you! Here are your details:\n"
                f"Name: {self.data['name']}\n"
                f"Email: {self.data['email']}\n"
                f"Phone: {self.data['phone_number']}\n"
                f"Appointment Date: {self.data['appointment_date']}"
            )
        elif self.state == "complete":
            return "Your information has been submitted successfully!"

    def store_in_database(self, session_id):
        form_entry = ConversationalFormData(
            form_id=session_id,
            name=self.data["name"],
            email=self.data["email"],
            phone_number=self.data["phone_number"],
            appointment_date=self.data["appointment_date"]
        )
        table = get_or_create_table("ConversationalFormData", ConversationalFormData, mode="create")
        table.add([form_entry.model_dump()])

    def validate_email(self, email):
        return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email)

    def validate_phone_number(self, phone_number):
        return re.match(r"^\+?[0-9]{10}$", phone_number)

# Knowledge Base Functionality
def summarize_chat_history(chat_history: str) -> dict:
    model = GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(
        f"You are an AI assistant tasked with summarizing previous queries and responses.\n{chat_history}"
    )
    return {
        "response": response.text.strip(),
        "input_token": 0,
        "output_token": 0,
        "total_token": 0
    }

def query_knowledge_base(kb_name: str, query: str, chat_history: str = ""):
    embedding_table_name = kb_name.replace(" ", "_").lower()
    kb_embeddings_table = get_or_create_table(embedding_table_name, KBEmbeddings, mode="create")
    query_embedding = generate_embeddings([query])[0]
    embeddings_data = kb_embeddings_table.to_pandas()
    embeddings_data["score"] = embeddings_data["vector"].apply(
        lambda vec: cosine_similarity(query_embedding, vec)
    )
    sorted_results = embeddings_data.sort_values(by="score", ascending=False)
    top_3_results = sorted_results.head(2)
    if top_3_results.empty:
        return {
            "response": "Sorry, I don't have an answer.",
            "chunk_indices": [],
            "scores": []
        }
    context = "\n".join(top_3_results["text"].tolist())
    chunk_indices = top_3_results["chunk_id"].tolist()
    scores = top_3_results["score"].tolist()
    full_context = (
        f"Knowledge Base Context:\n{context}\n"
        f"Question: {query}"
    )
    model = GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(full_context)
    return {
        "response": response.text.strip(),
        "kb_name": kb_name,
        "chunk_indices": chunk_indices,
        "scores": scores,
        "input_token": 0,
        "output_token": 0,
        "total_token": 0
    }

# Background task to process uploaded files
def process_uploaded_files(kb_name: str, uploaded_file_urls: List[str], task_id: str):
    try:
        task_status_table = get_or_create_table("task_status", TaskStatus, mode="create")
        task_status_table.update(
            where=f"task_id = '{task_id}'",
            values={"status": "pending"}
        )
        full_text = ""
        for file_path in uploaded_file_urls:
            ext = os.path.splitext(file_path)[1].lower()
            try:
                extracted_text = extract_text(file_path, ext)
                full_text += extracted_text + "\n"
            except Exception as e:
                logging.error(f"Failed to extract text from {file_path}: {str(e)}")
        chunks = split_text_into_chunks(full_text)
        embeddings = generate_embeddings(chunks)
        kb_metadata_table = get_or_create_table("kb_metadata", KBMetadata, mode="create")
        existing_kb = kb_metadata_table.to_pandas()
        existing_kb_names = existing_kb["kb_name"].tolist()
        if kb_name in existing_kb_names:
            kb_id = existing_kb[existing_kb["kb_name"] == kb_name]["kb_id"].iloc[0]
        else:
            kb_id = str(uuid4())
            kb_metadata_table.add([{
                "kb_id": kb_id,
                "kb_name": kb_name,
                "uploaded_file_url": ", ".join(uploaded_file_urls)
            }])
        embedding_table_name = kb_name.replace(" ", "_").lower()
        kb_embeddings_table = get_or_create_table(embedding_table_name, KBEmbeddings, mode="create")
        embeddings_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embeddings_data.append({
                "chunk_id": str(uuid4()),
                "kb_id": kb_id,
                "text": chunk,
                "vector": embedding
            })
        kb_embeddings_table.add(embeddings_data)
        task_status_table.update(
            where=f"task_id = '{task_id}'",
            values={"status": "success"}
        )
    except Exception as e:
        task_status_table.update(
            where=f"task_id = '{task_id}'",
            values={"status": "failed"}
        )

# Split text into chunks
def split_text_into_chunks(text: str, max_length: int = 500) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=50)
    return splitter.split_text(text)

# FastAPI endpoints
@app.post("/callback")
async def chat(session_id: str = Body(default=None), user_input: str = Body(default=None)):
    try:
        # Log the incoming request data
        logging.info(f"Received request with session_id={session_id}, user_input={user_input}")
        # Step 1: Validate the incoming session_id
        if not session_id:
            # Generate a new session_id only if none is provided
            session_id = generate_session_id()
            logging.info(f"Generated new session_id: {session_id}")
        # Step 2: Check if the session_id exists in the sessions dictionary
        if session_id not in sessions:
            logging.warning(f"Session ID {session_id} not found in sessions. Creating a new session.")
            sessions[session_id] = ConversationalForm()
            response = sessions[session_id].process_input("", session_id)
        else:
            logging.info(f"Continuing existing session for session_id: {session_id}")
            response = sessions[session_id].process_input(user_input, session_id)
        # Step 3: Return the session_id and response
        logging.info(f"Returning response: {response}")
        return {"session_id": session_id, "response": response}
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": "An unexpected error occurred."}

@app.post("/upload/")
async def upload_files_and_create_kb(kb_name: str = Form(...), files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    os.makedirs("uploads", exist_ok=True)
    uploaded_file_urls = []
    for file in files:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
            uploaded_file_urls.append(file_path)
    task_status_table = get_or_create_table("task_status", TaskStatus, mode="create")
    task_id = str(uuid4())
    task_status_table.add([{
        "task_id": task_id,
        "status": "pending"
    }])
    background_tasks.add_task(process_uploaded_files, kb_name, uploaded_file_urls, task_id)
    return JSONResponse(content={"message": "Uploaded and knowledge base is building", "task_id": task_id})

@app.get("/kb_metadata/")
async def list_knowledge_bases():
    kb_metadata_table = get_or_create_table("kb_metadata", KBMetadata, mode="create")
    kb_list = kb_metadata_table.to_pandas().to_dict(orient="records")
    return JSONResponse(content=kb_list)

@app.post("/newchat/")
async def new_chat(kb_name: str = Form(...)):
    chat_lookup_table = get_or_create_table("chat_lookup", ChatLookUp, mode="create")
    conv_id = str(uuid4())
    chat_lookup_table.add([{
        "conv_id": conv_id,
        "kb_name": kb_name
    }])
    chat_history_table_name = f"{conv_id}"
    get_or_create_table(chat_history_table_name, ChatTable, mode="create")
    return {"conv_id": conv_id}

@app.post("/chat/")
async def chat_with_kb(conv_id: str = Form(...), user_input: str = Form(...)):
    chat_lookup_table = get_or_create_table("chat_lookup", ChatLookUp, mode="create")
    chat_lookup_data = chat_lookup_table.to_pandas()
    if conv_id not in chat_lookup_data["conv_id"].tolist():
        return JSONResponse(status_code=404, content={"error": "Conversation ID not found."})
    kb_name = chat_lookup_data[chat_lookup_data["conv_id"] == conv_id]["kb_name"].iloc[0]
    chat_history_table_name = f"{conv_id}"
    chat_history_table = get_or_create_table(chat_history_table_name, ChatTable, mode="create")
    chat_history_data = chat_history_table.to_pandas()
    chat_history_prompt = ""
    for _, row in chat_history_data.iterrows():
        chat_history_prompt += f"User: {row['query']}\nAssistant: {row['response']}\n"
    try:
        result = query_knowledge_base(
            kb_name,
            user_input,
            chat_history=chat_history_prompt
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    chunk_ids = result.get("chunk_indices", [])
    scores = result.get("scores", [])
    chunk_texts = []
    input_token = result.get("input_token")
    output_token = result.get("output_token")
    total_token = result.get("total_token")
    if chunk_ids:
        embedding_table_name = kb_name.replace(" ", "_").lower()
        kb_embeddings_table = get_or_create_table(embedding_table_name, KBEmbeddings, mode="create")
        for chunk_id in chunk_ids:
            chunk_data = kb_embeddings_table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_pandas()
            if not chunk_data.empty:
                chunk_texts.append(chunk_data["text"].iloc[0])
            else:
                chunk_texts.append("")
    if not result["response"]:
        result["response"] = "Sorry, I don't have an answer."
        chunk_ids = []
        scores = []
        chunk_texts = []
    ct_id = str(uuid4())
    chat_history_table.add([{
        "ct_id": ct_id,
        "query": user_input,
        "response": result["response"],
        "chunk_ids": chunk_ids if chunk_ids else [],
        "similarity_scores": scores if scores else [],
        "chunk_texts": chunk_texts if chunk_texts else []
    }])
    chat_usage_table_name = f"chat_usage_{conv_id}"
    chat_usage_table = get_or_create_table(chat_usage_table_name, ChatUsage, mode="create")
    chat_usage_table.add([{
        "chu_id": str(uuid4()),
        "ct_id": ct_id,
        "input_token": input_token,
        "output_token": output_token,
        "total_token": total_token
    }])
    chat_history_table = get_or_create_table("chat_history", ChatHistory, mode="create")
    existing_entry = chat_history_table.to_pandas()
    if conv_id in existing_entry["conv_id"].tolist():
        chat_history_table.update(
            where=f"conv_id = '{conv_id}'",
            values={"title": user_input}
        )
    else:
        chat_history_table.add([{
            "ch_id": str(uuid4()),
            "conv_id": conv_id,
            "title": user_input
        }])
    return JSONResponse(content={
        "response": result["response"],
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts
    })

@app.get("/chathistory/")
async def get_chat_history():
    chat_history_table = get_or_create_table("chat_history", ChatHistory, mode="create")
    chat_history_data = chat_history_table.to_pandas()
    chat_history_list = chat_history_data[["conv_id", "title"]].to_dict(orient="records")
    return JSONResponse(content=chat_history_list)

@app.get("/chathistory/{conv_id}")
async def get_chat_history_by_conv_id(conv_id: str):
    chat_history_table_name = f"{conv_id}"
    chat_history_table = get_or_create_table(chat_history_table_name, ChatTable, mode="create")
    try:
        chat_history_data = chat_history_table.to_pandas()
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to fetch chat history: {str(e)}"},
            status_code=500
        )
    chat_history_list = []
    for _, row in chat_history_data.iterrows():
        chunk_ids = row["chunk_ids"].tolist() if isinstance(row["chunk_ids"], np.ndarray) else row["chunk_ids"]
        similarity_scores = row["similarity_scores"].tolist() if isinstance(row["similarity_scores"], np.ndarray) else row["similarity_scores"]
        chunk_texts = row.get("chunk_texts", [])
        if isinstance(chunk_texts, np.ndarray):
            chunk_texts = chunk_texts.tolist()
        chat_history_list.append({
            "query": row["query"],
            "response": row["response"],
            "chunk_ids": chunk_ids,
            "similarity_scores": similarity_scores,
            "chunk_texts": chunk_texts
        })
    return JSONResponse(content=chat_history_list)

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    task_status_table = get_or_create_table("task_status", TaskStatus, mode="create")
    task_data = task_status_table.to_pandas()
    if task_id not in task_data["task_id"].tolist():
        return JSONResponse(status_code=404, content={"error": "Task ID not found."})
    task = task_data[task_data["task_id"] == task_id].iloc[0]
    return JSONResponse(content={
        "task_id": task["task_id"],
        "status": task["status"],
    })