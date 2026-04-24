import streamlit as st
import requests
import re

API_URL = "http://localhost:8000"   # use "http://fastapi:8000" for Docker

# API_URL = "http://fastapi:8000" # Use this when running in Docker, as "localhost" would refer to the container itself

def clean_answer(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

st.set_page_config(page_title="Meeting Analyst", layout="wide")
st.title("📝 Meeting Transcript Q&A")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []   # each: {"role": str, "content": str, "sources": list}

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(f"{i}. **{src.get('speaker')}** at `{src.get('timestamp_str')}` – {src.get('source_file')}")

# Chat input
if prompt := st.chat_input("Ask about the meetings..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"question": prompt}, timeout=30).json()
                answer = clean_answer(resp["answer"])
                sources = resp.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"{i}. **{src.get('speaker')}** at `{src.get('timestamp_str')}` – {src.get('source_file')}")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})