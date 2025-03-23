# Copyright (c) 2025 NGSS Phenomena, LLC. All Rights Reserved.
# Author: T.J. McKenna
# Proprietary and confidential.
# See LICENSE file for licensing details.

import streamlit as st
import json
from app.utils.response_generation import ResponseGenerator

# Load curriculum metadata
@st.cache_data
def load_metadata():
    with open("data/metadata/curriculum_metadata.json") as f:
        return json.load(f)

metadata = load_metadata()
generator = ResponseGenerator()

st.title("🧪 NGSS Curriculum Coach")

st.markdown("Ask me about lessons, SEPs, models, or assessments — I'm here to help!")

# Session state to hold chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("Ask your curriculum question here...")

# Display past messages
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if query:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Simulate simple retrieval: grab top 2 documents from metadata
    # (replace this with real RAG logic later)
    fake_docs = []
    for item in metadata["files"][:2]:
        fake_docs.append(type("Doc", (), {
            "page_content": "Example content from: " + item["title"],
            "metadata": item
        })())

    # Simulate intent detection (very basic keyword check)
    lower_query = query.lower()
    if any(word in lower_query for word in ["sep", "practice"]):
        intent = "sep_identifier"
    elif "model" in lower_query:
        intent = "model_analyzer"
    elif "assessment" in lower_query:
        intent = "assessment_mapper"
    elif "lesson" in lower_query or "dci" in lower_query or "phenomenon" in lower_query:
        intent = "lesson_finder"
    else:
        intent = "general_guidance"

    # Generate coach response
    response = generator.generate_response(
        query=query,
        intent=intent,
        retrieved_docs=fake_docs,
        user_context={"grade_level": 6},
        chat_history=st.session_state.chat_history
    )

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
