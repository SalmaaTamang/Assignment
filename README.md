# Assignment
## Chat with docs App with Conservational Form for booking appointment
### Overview  

### The application integrates various technologies such as: 

    FastAPI : For building RESTful APIs.
    LanceDB : A vector database for storing embeddings and metadata.
    Gemini API : For generating responses to user queries.
    SentenceTransformer : For generating text embeddings.
    PyPDF2, Docx, PIL, pytesseract : For extracting text from PDF, DOCX, images, and plain text files.
    LangChain : For splitting text into chunks.
     

### The system supports: 

    Uploading files and creating knowledge bases.
    Querying the knowledge base using natural language.
    Managing chat history and usage statistics.
    Collecting user information via a conversational form.


### This Streamlit application provides a user-friendly interface for interacting with a FastAPI backend. The app allows users to:

    Upload files to a knowledge base
    
    Engage in chatbot conversations
    
    View and manage chat history
    
    Book appointments
    
#### Features
    
    1. Landing Page
    
    A simple welcome page introducing the application.
    
    2. Create Ingestion
    
    Users can upload files (PDF, DOCX, TXT, JPG, PNG) to create a knowledge base.
    
    The application submits files to the FastAPI backend for processing.
    
    The ingestion process is tracked, and the status is updated in real-time.
    
    3. Chat with Docs
    
    Users can chat with the system based on knowledge bases.
    
    The sidebar displays available knowledge bases for selection.
    
    A chat history section is available to view past conversations.
    
    Users can start a new chat session.
    
    Responses are provided along with chunked text sources for reference.
    
    4. Book an Appointment
    
    Users can initiate a chatbot-based appointment booking process.
    
    Conversations are managed using session IDs.
    
    The chatbot interacts with users to facilitate appointment scheduling.


    
     

### Database Structure  

The application uses LanceDB to store data in multiple tables. Each table is represented by a Pydantic model, and the relationships between tables are managed using foreign keys. 
Tables and Their Descriptions  

    TaskStatus  
        Tracks the status of background tasks (e.g., file processing).
        Fields :
            task_id: Unique identifier for the task.
            status: Current status (pending, success, failed).
             
         

    KBMetadata  
        Stores metadata about knowledge bases.
        Fields :
            kb_id: Unique identifier for the knowledge base.
            kb_name: Name of the knowledge base.
            uploaded_file_url: URLs of the uploaded files.
             
         

    KBEmbeddings  
        Stores text chunks and their embeddings for each knowledge base.
        Fields :
            chunk_id: Unique identifier for the chunk.
            kb_id: Foreign key referencing the knowledge base.
            text: Text content of the chunk.
            vector: Embedding vector (size 384).
             
         

    ChatLookUp  
        Maps conversation IDs to knowledge base names.
        Fields :
            conv_id: Unique identifier for the conversation.
            kb_name: Name of the knowledge base associated with the conversation.
             
         

    ChatTable  
        Stores chat history for each conversation.
        Fields :
            ct_id: Unique identifier for the chat entry.
            query: User query.
            response: AI-generated response.
            chunk_ids: IDs of the chunks used to generate the response.
            chunk_texts: Texts of the chunks.
            similarity_scores: Similarity scores of the chunks.
             
         

    ChatHistory  
        Stores titles for conversations.
        Fields :
            ch_id: Unique identifier for the chat history entry.
            conv_id: Foreign key referencing the conversation.
            title: Title of the conversation.
             
         

    ChatUsage  
        Tracks token usage for each chat interaction.
        Fields :
            chu_id: Unique identifier for the usage entry.
            ct_id: Foreign key referencing the chat entry.
            input_token: Number of input tokens.
            output_token: Number of output tokens.
            total_token: Total tokens used.
             
         

    ConversationalFormData  
        Stores user-submitted information from the conversational form.
        Fields :
            form_id: Unique identifier for the form entry.
            name: User's name.
            email: User's email address.
            phone_number: User's phone number.
            appointment_date: Preferred appointment date.
             
         
     

### Models Used  
Embedding Model  

    SentenceTransformer ('all-MiniLM-L6-v2') :
        Generates 384-dimensional embeddings for text chunks.
        Used for similarity search in the knowledge base.
         
     

Generative Model  

    Gemini API ('gemini-2.0-flash-exp') :
        Generates responses to user queries based on context from the knowledge base.
         
     

Text Splitter  

    RecursiveCharacterTextSplitter :
        Splits extracted text into smaller chunks for embedding generation.
         
     

Text Extraction Libraries  

    PyPDF2 : Extracts text from PDF files.
    Docx : Extracts text from DOCX files.
    PIL + pytesseract : Extracts text from images.
    Plain Text Reader : Reads text from .txt files.
     

Information Flow  

    File Upload and Knowledge Base Creation : 
        Users upload files (PDF, DOCX, images, etc.) via the /upload/ endpoint.
        The system extracts text from the files and splits it into chunks.
        Embeddings are generated for each chunk and stored in the KBEmbeddings table.
        Metadata about the knowledge base is stored in the KBMetadata table.
         

    Querying the Knowledge Base : 
        Users send natural language queries via the /chat/ endpoint.
        The system retrieves relevant chunks from the knowledge base using cosine similarity.
        A response is generated using the Gemini API and returned to the user.
         

    Conversational Form : 
        Users interact with the /callback endpoint to provide their name, email, phone number, and preferred appointment date.
        The system validates the input and stores it in the ConversationalFormData table.
         

    Chat History and Usage Tracking : 
        Chat interactions are stored in the ChatTable.
        Token usage is tracked in the ChatUsage table.
        Conversation titles are stored in the ChatHistory table.
         
     

### API Endpoints  
1. File Upload and Knowledge Base Creation  

    Endpoint : /upload/
    Method : POST
    Parameters :
        kb_name: Name of the knowledge base.
        files: List of uploaded files.
         
    Response :
        message: Status message.
        task_id: ID of the background task.
         
     

2. List Knowledge Bases  

    Endpoint : /kb_metadata/
    Method : GET
    Response :
        List of knowledge bases with their metadata.
         
     

3. Start a New Chat  

    Endpoint : /newchat/
    Method : POST
    Parameters :
        kb_name: Name of the knowledge base.
         
    Response :
        conv_id: Unique identifier for the conversation.
         
     

4. Chat with Knowledge Base  

    Endpoint : /chat/
    Method : POST
    Parameters :
        conv_id: Conversation ID.
        user_input: User query.
         
    Response :
        response: AI-generated response.
        chunk_ids: IDs of the chunks used.
        chunk_texts: Texts of the chunks.
         
     

5. Get Chat History  

    Endpoint : /chathistory/
    Method : GET
    Response :
        List of conversations with their titles.
         
     

6. Get Chat History by Conversation ID  

    Endpoint : /chathistory/{conv_id}
    Method : GET
    Response :
        Detailed chat history for the specified conversation.
         
     

7. Check Task Status  

    Endpoint : /task_status/{task_id}
    Method : GET
    Response :
        task_id: Task ID.
        status: Current status of the task.
         
     

8. Conversational Form  

    Endpoint : /callback
    Method : POST
    Parameters :
        session_id: Session ID (optional).
        user_input: User input.
         
    Response :
        session_id: Updated session ID.
        response: Next question or confirmation message.
         
     
             
         
     
