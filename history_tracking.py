
import os
from flask import Flask, request, jsonify
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['OPEN_API_KEY'] = ''

mysql_uri = "mysql+mysqlconnector://root:1999@127.0.0.1:3307/healthq"
db = SQLDatabase.from_uri(mysql_uri)

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

def get_schema(db):
    """Fetch the database schema."""
    schema = db.get_table_info()
    return schema

api_key = os.environ['OPEN_API_KEY']
llm = ChatOpenAI(api_key=api_key)

sql_chain = (
    RunnablePassthrough.assign(schema=lambda x: get_schema(db))
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

template_response = """Based on the table schema below, question, sql query, and sql response, write a natural language response. Additionally, explain why this response was generated, including why the specific ClaimID (e.g., ClaimID = 1) was chosen and how it relates to the user's question. Structure your response as follows:

Natural Language Response: <your response here>
Explanation: <your explanation here, focusing on why this response was generated, why the specific ClaimID (e.g., ClaimID = 1) was chosen, and how it relates to the user's question>

Schema:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

prompt_response = ChatPromptTemplate.from_template(template_response)

def run_query(query):
    """Execute a SQL query and return the result."""
    return db.run(query)

def extract_response_parts(response):
    """Extract natural language response and explanation from LLM response."""
    content = response.content if hasattr(response, 'content') else str(response)
    
    if "Explanation:" in content:
        parts = content.split("Explanation:")
        natural_language = parts[0].replace("Natural Language Response:", "").strip()
        explanation = parts[1].strip()
    else:
        natural_language = content.replace("Natural Language Response:", "").strip()
        explanation = "No explanation provided."
    
    return natural_language, explanation

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(schema=lambda x: get_schema(db),
        response=lambda vars: run_query(vars["query"]),
    )
    | prompt_response
    | llm
)

explanation_chain = (
    RunnablePassthrough.assign(
        schema=lambda x: get_schema(db),
        query=lambda x: x["query"],
        response=lambda x: x["response"],
        natural_language_response=lambda x: x["natural_language_response"],
        question=lambda x: x["question"]
    )
    | prompt_response
    | llm
)

app = Flask(__name__)

conversation_history = []

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(question1, question2):
    """Calculate cosine similarity between two questions."""
    embeddings = embedding_model.encode([question1, question2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'Response': 'SQL Query API is Running'})

# Query endpoint
@app.route('/query', methods=['POST'])
def query_db():
    """Endpoint to handle user questions and return SQL query, natural language response, and explanation."""
    try:
        logger.debug(f"Incoming request: {request.json}")

        user_question = request.json.get("prompt")
        history = request.json.get("history", [])  

        if not user_question:
            logger.error("No question provided in the request")
            return jsonify({"response": "No question provided"}), 200

        logger.debug(f"Processing question: {user_question}")
        logger.debug(f"History: {history}")

        is_follow_up = False
        combined_question = user_question
        if history:
            last_interaction = history[-1]
            last_question = last_interaction.get("question")
            last_response = last_interaction.get("natural_language_response")

            similarity_score = calculate_similarity(user_question, last_question)
            logger.debug(f"Similarity score: {similarity_score}")

            if similarity_score > 0.7: 
                is_follow_up = True
                combined_question = f"{last_question}, and {user_question}"

        if is_follow_up:
            logger.debug(f"Follow-up question detected. Combined question: {combined_question}")

        sql_query = sql_chain.invoke({"question": combined_question})
        sql_response = run_query(sql_query)

        full_response = full_chain.invoke({"question": combined_question, "query": sql_query, "response": sql_response})
        
        if "Explanation:" in full_response.content:
            response_parts = full_response.content.split("Explanation:")
            natural_language_response = response_parts[0].replace("Natural Language Response:", "").strip()
            explanation = response_parts[1].strip()
        else:
            natural_language_response = full_response.content.replace("Natural Language Response:", "").strip()
            explanation = "No explanation provided."

        explanation_result = explanation_chain.invoke({
            "query": sql_query,
            "response": sql_response,
            "natural_language_response": natural_language_response,
            "question": combined_question  
        })
        detailed_explanation = explanation_result.content  

        conversation_history.append({
            "question": combined_question,
            "query": sql_query,
            "response": sql_response,
            "natural_language_response": natural_language_response,
            "explanation": detailed_explanation
        })

        logger.debug(f"Generated SQL query: {sql_query}")
        logger.debug(f"Generated natural language response: {natural_language_response}")
        logger.debug(f"Generated explanation: {detailed_explanation}")

        return jsonify({
            'response': {
            "input": user_question,
            "output": natural_language_response,
            "explanation": detailed_explanation,
            "history": conversation_history  
            }
        })
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({"response": f"An error occurred: {str(e)}"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
