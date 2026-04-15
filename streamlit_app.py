import streamlit as st
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.chat_history import get_recent_turns, clear_memory, store_turn

API_URL = "http://localhost:8000"
SESSION_ID = "default_user_session"

def init_ui():
    st.set_page_config(page_title="ContextRAG Support", page_icon="💬", layout="centered")
    st.markdown("<h1 style='color: #8B5A2B;'>Chat with ContextRAG</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_cleared" not in st.session_state:
        st.session_state.memory_cleared = False
    if "quick_prompt" not in st.session_state:
        st.session_state.quick_prompt = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("<h3 style='color: #8B5A2B;'>⚙️ Settings & Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
        if st.button("Process Document"):
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                        response = requests.post(f"{API_URL}/upload", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"{data['message']}. Indexed {data['chunks_indexed']} chunks.")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")
            else:
                st.warning("Please upload a file first.")
                
        st.divider()
        st.header("Search Settings")
        mode_mapping = {
            "🔵 Hybrid (Recommended)": "hybrid",
            "🟣 Semantic only": "semantic",
            "🟡 Keyword only": "keyword"
        }
        selected_mode_label = st.radio("Search Mode", list(mode_mapping.keys()))
        selected_mode = mode_mapping[selected_mode_label]
        
        st.divider()
        st.header("Memory Management")
        if st.button("🧹 Clear Memory"):
            clear_memory(SESSION_ID)
            st.session_state.memory_cleared = True
            st.success("Session memory wiped.")

    # Memory state logic visualizers
    memory_disabled = False
    if selected_mode == "keyword":
        memory_disabled = True
        st.info("💡 Keyword mode focuses purely on exact text matching and disables multi-turn Conversation Memory.")

    if st.session_state.memory_cleared:
        st.caption("Memory cleared — Gemini no longer sees prior turns. ContextRAG operates in read-only visual mode for previous chats.")

    recent_turns = get_recent_turns(SESSION_ID, n=4)
    if not memory_disabled and len(recent_turns) >= 2 and not st.session_state.memory_cleared:
        st.caption("💬 Using conversation context")

    # Display chat history from session state
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View sources", expanded=False):
                    for idx, src in enumerate(message["sources"]):
                        chunk_text = src.get("chunk", "")
                        breakdown = src.get("breakdown", {})
                        st.markdown(f"**Source {idx+1}:**\n{chunk_text}")
                        if breakdown:
                            st.caption(f"Semantic: {breakdown.get('semantic', 0)} | Keyword: {breakdown.get('keyword', 0)} | Combined: {breakdown.get('combined', 0)}")
                        elif "score" in src:
                            st.caption(f"Score: {src['score']}")
                        st.divider()

    # Empty State Greeting & Action Buttons
    if len(st.session_state.messages) == 0:
        st.markdown("<h2 style='text-align: center; color: #8B5A2B;'>Good Afternoon!</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Let's get you some help. Please select an option so I can assist you with your document.</p>", unsafe_allow_html=True)
        st.write("")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Summarize Document", use_container_width=True, help="Get a quick summary of the uploaded text."):
                st.session_state.quick_prompt = "Summarize the key points of the uploaded document."
                st.rerun()
        with col2:
            if st.button("🔍 Find Important Concepts", use_container_width=True, help="Extract main ideas or entities."):
                st.session_state.quick_prompt = "What are the most important concepts and entities discussed in this document?"
                st.rerun()
                
        st.divider()

    # Input Box logic
    # Check if a quick prompt was clicked, otherwise use chat input
    user_input = st.chat_input("Enter a question or response...")
    
    # Handle quick_prompt injection
    if st.session_state.get("quick_prompt"):
        user_input = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    if user_input:
        st.session_state.memory_cleared = False
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            sources_list = []
            
            with st.spinner("Gemini is exploring context..."):
                try:
                    # Gather context history dynamically based on memory constraints
                    send_history = []
                    if not memory_disabled:
                        send_history = get_recent_turns(SESSION_ID, n=4)
                        
                    payload = {
                        "question": user_input,
                        "chat_history": send_history,
                        "mode": selected_mode
                    }
                    
                    response = requests.post(f"{API_URL}/query", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer provided.")
                        sources_list = data.get("sources", [])
                        
                        # Save successful history turn securely
                        if not memory_disabled:
                            store_turn(SESSION_ID, user_input, answer, sources_list)
                    else:
                        answer = f"Error querying backend: {response.json().get('detail', 'Unknown error')}"
                except Exception as e:
                    answer = f"Failed to connect to backend API: {e}"
                
            message_placeholder.markdown(answer)
            if sources_list:
                with st.expander("View sources", expanded=False):
                    for idx, src in enumerate(sources_list):
                        chunk_text = src.get("chunk", "")
                        breakdown = src.get("breakdown", {})
                        st.markdown(f"**Source {idx+1}:**\n{chunk_text}")
                        if breakdown:
                            st.caption(f"Semantic: {breakdown.get('semantic', 0)} | Keyword: {breakdown.get('keyword', 0)} | Combined: {breakdown.get('combined', 0)}")
                        elif "score" in src:
                            st.caption(f"Score: {src['score']}")
                        st.divider()
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_list
            })

if __name__ == "__main__":
    if "test" in sys.argv:
        print("Streamlit app syntax ok.")
        sys.exit(0)
    else:
        init_ui()
