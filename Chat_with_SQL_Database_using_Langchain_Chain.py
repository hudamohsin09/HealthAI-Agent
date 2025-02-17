import os
from flask import Flask, request, jsonify
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set your OpenAI API key here
os.environ['OPEN_API_KEY'] = ''

# MySQL database URI
mysql_uri = "mysql+mysqlconnector://root:1999@127.0.0.1:3307/healthq"
db = SQLDatabase.from_uri(mysql_uri)

# Template for generating SQL queries
template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

def get_schema(db):
    """Fetch the database schema."""
    schema = db.get_table_info()
    return schema

# Initialize the ChatOpenAI model with the API key
api_key = os.environ['OPEN_API_KEY']
llm = ChatOpenAI(api_key=api_key)

# SQL chain to generate queries
sql_chain = (
    RunnablePassthrough.assign(schema=lambda x: get_schema(db))
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Full chain to generate natural language responses
template_response = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template_response)

def run_query(query):
    """Execute a SQL query and return the result."""
    return db.run(query)

# Full chain to process user questions
full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(schema=lambda x: get_schema(db),
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm
)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'Response': 'SQL Query API is Running'})

@app.route('/query', methods=['POST'])
def query_db():
    """Endpoint to handle user questions and return SQL query results."""
    try:
        # Log the incoming request
        logger.debug(f"Incoming request: {request.json}")

        # Extract the user's question from the request
        user_question = request.json.get("question")
        if not user_question:
            logger.error("No question provided in the request")
            return jsonify({"Response": "No question provided"}), 400

        # Log the user's question
        logger.debug(f"Processing question: {user_question}")

        # Invoke the full chain to process the question
        response = full_chain.invoke({"question": user_question})

        # Log the response
        logger.debug(f"Generated response: {response}")

        # Return the response in JSON format
        return jsonify({"Response": response.content})
    except Exception as e:
        # Log the error
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"Response": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)