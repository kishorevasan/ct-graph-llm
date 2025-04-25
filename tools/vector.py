import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Create the Neo4jVector
ct_trial_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph = graph,
    index_name ='ctDescription',
    node_label='ClinicalTrial',
    text_node_property = 'title',
    embedding_node_property = 'plotEmbedding',
    retrieval_query = """
    RETURN node.title AS text, score,
    {
        id: node.id,
        source: 'https://clinicaltrials.gov/study/'+node.id,
        summary: node.summary
    } AS metadata
    """
)

# Create the retriever
retriever = ct_trial_vector.as_retriever()

# Create the prompt
instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [("system", instructions),
     ("human", "{input}")
     ]
)

# Create the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
ct_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Create a function to call the chain
def get_ct_description(input):
    return ct_retriever.invoke({"input":input})
