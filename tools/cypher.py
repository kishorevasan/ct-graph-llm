import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher queries about clinical trials.
Convert the user's question based on the schema.
When you respond to a query, do not return vector embeddings or any internal representations as raw data.
Instead, provide clear, concise, and human-readable answers that explain or summarize the results.
If requested to return embeddings or vectors, kindly clarify that they cannot be provided and suggest an alternative solution.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Do not use primary keys to filter nodes, only use WHERE and CONTAINS to search strings.

Example Cypher Statements:

1. To find number of clinical trials in the US:
```
MATCH (ct:ClinicalTrial)-[:IN_COUNTRY]->(c:Country {{name: "United States"}})
RETURN COUNT(ct) AS num_clinical_trials
```

2. To find the sponsors of Breast Cancer Trials
```
MATCH (s:Sponsor)<-[:SPONSORED_BY]-(ct:ClinicalTrial)-[:HAS_CONDITION]->(cond:Condition)
WHERE cond.name CONTAINS "Breast Cancer"
RETURN DISTINCT s.name AS sponsor
```

3. To find the locations of clinical trials by a specific sponsor and condition:
```
MATCH (s:Sponsor)<-[:SPONSORED_BY]-(ct:ClinicalTrial)-[:HAS_CONDITION]->(cond:Condition), (ct)-[:IN_CITY]->(c:City)
WHERE s.name CONTAINS "AstraZeneca" AND cond.name CONTAINS "Breast Cancer"
RETURN c.name
```

4. To get the summary of ongoing trials for a specific disease:
```
MATCH (ct:ClinicalTrial)-[:HAS_CONDITION]->(c:Condition)
WHERE ct.status IN ['RECRUITING','ACTIVE_NOT_RECRUITING','ENROLLING_BY_INVITATION','NOT_YET_RECRUITING'] AND c.name CONTAINS "Obesity"
RETURN ct.summary
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
