import os
import re
import streamlit as st
import requests

# Fallback: localhost for local dev, fastapi service name inside Docker
API_URL = os.getenv("API_URL", "http://localhost:8000")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_answer(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


st.set_page_config(page_title="Meeting Analyst", layout="wide")
st.title("📝 Meeting Transcript Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role", "content", "sources"}

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.write(
                        f"{i}. **{src.get('speaker', 'Unknown')}** "
                        f"at `{src.get('timestamp_str', '00:00')}` "
                        f"— {src.get('source_file', 'Unknown')}"
                        + (f" · *{src.get('topic', '')}*" if src.get("topic") else "")
                    )

# Chat input
if prompt := st.chat_input("Ask about the meetings..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "chat_history": history_payload,
                        "hybrid": True,
                        "bm25_weight": 0.3,
                    },
                    timeout=60,
                ).json()

                answer = clean_answer(resp.get("answer", "No answer returned."))
                sources = resp.get("sources", [])
                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.write(
                                f"{i}. **{src.get('speaker', 'Unknown')}** "
                                f"at `{src.get('timestamp_str', '00:00')}` "
                                f"— {src.get('source_file', 'Unknown')}"
                                + (f" · *{src.get('topic', '')}*" if src.get("topic") else "")
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
