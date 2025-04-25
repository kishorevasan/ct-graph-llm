import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about clinical trials.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statements:

1. To find number of clinical trials in the US:
```
MATCH (ct:ClinicalTrial)-[:IN_COUNTRY]->(c:Country {{name: "United States"}})
RETURN COUNT(ct)
```

2. Get diabetes clinical trials:
```
MATCH (ct:ClinicalTrial)
WHERE ct.title CONTAINS 'diabetes'
RETURN ct.id, ct.title, ct.summary
```

Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate(template = CYPHER_GENERATION_TEMPLATE,
                 input_variables = ['schema','question'])

# Create the Cypher QA chain
cypher_ct_chain = GraphCypherQAChain.from_llm(
    llm,
    graph = graph,
    verbose = True,
    cypher_prompt = cypher_prompt,
    allow_dangerous_requests = True
)
