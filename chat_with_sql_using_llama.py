import os
import mysql.connector
from flask import Flask, request, jsonify
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define MySQL database URI
mysql_uri = "mysql+mysqlconnector://root:1999@127.0.0.1:3307/healthq"
db = SQLDatabase.from_uri(mysql_uri)

# Function to fetch database schema information
def fetch_database_schema():
    # Establish a connection to the MySQL database
    conn = mysql.connector.connect(
        user="root",
        password="1999",
        host="127.0.0.1",
        port=3307,
        database="healthq"
    )
    cursor = conn.cursor()
    
    # Retrieve all table names
    cursor.execute("SHOW TABLES;")
    tables = [table[0] for table in cursor.fetchall()]
    
    # Fetch column details for each table
    schema_info = ""
    for table in tables:
        cursor.execute(f"DESCRIBE {table};")
        columns = cursor.fetchall()
        schema_info += f"Table `{table}` has columns: " + ", ".join([f"{col[0]} ({col[1]})" for col in columns]) + "\n"
    
    # Close the cursor and connection
    cursor.close()
    conn.close()
    
    return schema_info

# Retrieve the database schema
schema_info = fetch_database_schema()

# Define the system message for the SQL assistant
system_message = f"""
You are a specialized SQL assistant designed to translate natural language questions into precise MySQL queries. 

Here is the database schema:
{schema_info}

Rules:
1. Return only a valid MySQL query in JSON format.
2. Ensure the query is syntactically correct and adheres to MySQL standards.
3. Do NOT generate queries for DELETE, UPDATE, or DROP operations.
4. Always use accurate table and column names as per the provided schema.

Output Format:
{{
    "sql_query": "<Generated SQL Query>"
}}
"""

# Initialize the Ollama model (Llama 3.2)
llm = ChatOllama(
    model="llama3.2",  # Replace with the correct model name if different
    messages=[{"role": "system", "content": system_message}],
    base_url="http://localhost:11434",  # Default Ollama base URL
    temperature=0  # Set temperature to 0 for deterministic outputs
)

# Create the SQL agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Initialize Flask app
app = Flask(__name__)

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"Response": "SQL Query API is Running!"})

# Endpoint to handle SQL queries
@app.route('/query', methods=['POST'])
def query_db():
    try:
        # Extract the user's question from the request
        user_prompt = request.json["question"]
        
        # Invoke the SQL agent to process the question
        response = agent_executor.invoke(user_prompt)
        
        # Return the agent's response
        return jsonify({"Response": response})
    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"Response": f"An error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)