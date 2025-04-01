# Assignment
## Chat with docs App with Conservational Form for booking appointment
Overview  

The application integrates various technologies such as: 

    FastAPI : For building RESTful APIs.
    LanceDB : A vector database for storing embeddings and metadata.
    Gemini API : For generating responses to user queries.
    SentenceTransformer : For generating text embeddings.
    PyPDF2, Docx, PIL, pytesseract : For extracting text from PDF, DOCX, images, and plain text files.
    LangChain : For splitting text into chunks.
     

The system supports: 

    Uploading files and creating knowledge bases.
    Querying the knowledge base using natural language.
    Managing chat history and usage statistics.
    Collecting user information via a conversational form.
     

Database Structure  

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
             
         
     
