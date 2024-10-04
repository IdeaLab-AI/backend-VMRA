import os
import uuid
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from backend import get_embeddings_vector

# Load environment variables from .env file (if exists)
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

# Initialize Flask app
app = Flask(__name__)

# Fetch environment variables for Azure Search and OpenAI
azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_service_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_chat_completions_deployment_name = os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME")

# Fetch environment variables for Cosmos DB
cosmos_db_uri = os.getenv("COSMOS_DB_URI")
cosmos_db_key = os.getenv("COSMOS_DB_PRIMARY_KEY")
database_name = os.getenv("COSMOS_DB_DATABASE_ID")
container_name = os.getenv("COSMOS_DB_CONTAINER_ID")

# Initialize the Azure Search client
credential = AzureKeyCredential(azure_search_service_admin_key)
search_client = SearchClient(
    endpoint=azure_search_service_endpoint,
    index_name=search_index_name,
    credential=credential
)

# Initialize Cosmos DB client
cosmos_client = CosmosClient(cosmos_db_uri, credential=cosmos_db_key)
database = cosmos_client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Function to retrieve conversation history from Cosmos DB
@app.route("/getchathistory", methods=["POST"])
def get_history():
    """
    Fetch user chat history from Cosmos DB.
    """
    data = request.json
    user_id = data.get('user_id')
    
    query = "SELECT * FROM c WHERE c.id=@id ORDER BY c.timestamp DESC"
    parameters = [{"name": "@id", "value": user_id}]

    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    print(items, "items")
    
    response = [{"req": item.get('user_message'), "res": item.get('bot_response')} for item in items]
    
    return jsonify(items[0]["history"])


def get_conversation_history(user_id, thread_id):
    """
    Get conversation history based on user_id and thread_id.
    """
    query = "SELECT * FROM c WHERE c.id=@id ORDER BY c.timestamp DESC"
    parameters = [{"name": "@id", "value": user_id}]
    
    # Execute query and return result
    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    return items

def format_memory(conversation_history):
    """
    Format the conversation history to be used as memory.
    """
    return "\n".join(
        [f"User: {item['req']}\nBot: {item['res']}" for item in conversation_history[-5:]]
    )

def get_llm_response(user_input, search_content, user_id, thread_id):
    """
    Get response from Azure OpenAI using previous conversation as memory.
    """
    conversation_history = get_conversation_history(user_id, thread_id)
    
    # Prepare memory from previous chats if available
    if conversation_history:
        history = conversation_history[0].get("history", {})
        thread_history = history.get(thread_id, {}).get("chat", [])
        memory = format_memory(thread_history)
    else:
        memory = ""

    # Debug prints (optional)
    print("Conversation History:", conversation_history)
    print("Formatted Memory:", memory)

    # Prepare messages for the LLM interaction
    messages = [
        {"role": "system", "content": """
        
        You are the most intelligent Vintage Motorcycle Repair Assistant, you will help with any concern about motorbike repairing. When a user comes to you, you will start by welcoming them and asking for details like the brand, model, year, and the specific mechanical problem they are facing (e.g., engine, transmission, ignition) one by one. Ask one question at a time, with a minimum of 20 words and a maximum of 50 words per question. Do not mention any bike names on your own; gather all necessary information from the user.
        
        Probe further to clarify the issue and any steps they have already taken. Offer step-by-step guidance based on original shop manuals, always prioritizing safety and clarity, especially for beginners. Include clickable page references to the manuals to allow users to consult the original information. Depending on the situation, offer schematics or drawings to help users visualize repairs. If they upload pictures of parts, assist in finding affordable replacement options online. Encourage professional help when the issue is too complex. If you have no answer regarding the user's problem, provide the support contact (support@VintageMotorcycleRepairAssistant.com) if needed.

        provide the solution in bullet point and specfiy steps to do, and provide
        
        Throughout the interaction, adapt your advice to the user’s skill level, ensuring instructions are both clear and helpful, while maintaining a friendly, empathetic tone.
        
        REMEMBER TO NOT GIVE ANY REFERENCE FROM THE RAG OR THE PDF WITHOUT TAKING THE USER'S DETAILS.
        Make answers more formatted and clear. Be professional with any question and answer. Before answering, ask the user in brief about their actual problem one by one before giving any solution. You have to ask multiple questions about the problem 'one by one' before answering them.
        
        Remember you will get the PDF for reference, but you have to use it or make decisions and give accurate answers to the user.
        
        Just introduce yourself one time in the first question. Do not introduce yourself again and again, unless the user asks.
        """},  # Detailed system instructions
    ]
    
    if memory:
        messages.append({"role": "user", "content": memory})

    # Add current user input and search results
    messages.append({"role": "user", "content": f"{user_input}\n\nSearch results:\n{search_content}"})

    # Call Azure OpenAI for response
    openai_client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2024-06-01"
    )
    
    response = openai_client.chat.completions.create(
        model=azure_openai_chat_completions_deployment_name,
        messages=messages
    )

    return response.choices[0].message.content

def save_conversation(user_id, thread_id, user_message, bot_response, images):
    """
    Save conversation history to Cosmos DB, either append to existing or create new entry.
    """
    existing_conversation = get_conversation_history(user_id, thread_id)
    oldHistory = existing_conversation[0]["history"]
    isThreadPresent = thread_id in oldHistory 
    if isThreadPresent:
        # Append to existing conversation
        old_chats = existing_conversation[0]["history"].get(thread_id, {}).get("chat", [])
        item = {
            'id': user_id,
            'history': {
                thread_id: {
                    'heading': user_message,  # Optionally use first message as heading
                    'chat': [
                        *old_chats,
                        {'req': user_message, 'res': bot_response, "img":images}
                    ]
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    else:
        # Create new conversation
        item = {
            'id': user_id,
            'history': {
                **oldHistory,
                thread_id: {
                    'heading': user_message,
                    'chat': [{'req': user_message, 'res': bot_response,"img":images}]
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    # Upsert the conversation to Cosmos DB
    container.upsert_item(item)

def search_with_vector(query):
    """
    Perform vector search using the user query and return search results.
    """
    embedding = get_embeddings_vector(query)
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="vector")

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["page_number", "page_content"]
    )

    search_results = []
    for idx, result in enumerate(results, start=1):
        search_results.append({
            "doc_ref": f"[doc{idx}]",
            "page_number": result.get('page_number'),
            "page_content": result.get('page_content'),
            "score": result.get('@search.score'),
            "image_path": f"output_images/{str(result.get('page_number')).lower()}.jpg"
        })

    return search_results

def format_search_content(search_results):
    """
    Format search results to be used as context for the LLM.
    """
    return "\n".join(
        [f"{res['doc_ref']} page {res['page_number']}: {res['page_content']}" for res in search_results]
    )

@app.route("/", methods=["GET"])
def welcome():
    """
    Welcome endpoint for the API.
    """
    return jsonify(message="Hello, welcome to the API :)")

@app.route('/ask', methods=['POST'])
def ask():
    """
    Endpoint to interact with the LLM. It processes user input and returns the AI response.
    """
    data = request.json
    user_id = data.get('user_id')
    thread_id = data.get('thread')
    user_input = data.get('question')

    # Validate inputs
    if not user_id or not thread_id or not user_input:
        return jsonify({"error": "Missing 'user_id', 'thread', or 'question' in request."}), 400

    try:
        # Perform vector search based on user input
        search_results = search_with_vector(user_input)

        # Format the search content for LLM
        formatted_content = format_search_content(search_results)

        # Get response from LLM
        response = get_llm_response(user_input, formatted_content, user_id, thread_id)


        # Attach image paths if the response exceeds 160 words
        word_count = len(response.split())
        images = [res['image_path'] for res in search_results] if word_count > 160 else []

        # Save conversation history
        save_conversation(user_id, thread_id, user_input, response , images)

        print(f"Word count: {word_count}, Images: {images}")


        return jsonify({
            'response': response,
            'images': images  
        })
    except Exception as e:
        # Log the exception details for debugging
        app.logger.error(f"Exception in /ask: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/history/<user_id>/<thread_id>', methods=['GET'])
def history(user_id, thread_id):
    """
    Endpoint to retrieve conversation history by user and thread.
    """
    try:
        conversation_history = get_conversation_history(user_id, thread_id)
        if not conversation_history:
            return jsonify({"error": "No conversation history found."}), 404

        history = conversation_history[0].get("history", {})
        thread_history = history.get(thread_id, {}).get("chat", [])
        response = [{"req": item.get('req'), "res": item.get('res')} for item in thread_history]

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Exception in /history: {str(e)}")
        return jsonify({"error": "An error occurred while fetching conversation history."}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
