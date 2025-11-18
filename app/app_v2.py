import streamlit as st
import urllib.parse
import os
from datetime import datetime

# ----------------------------
# Config & Setup
# ----------------------------
st.set_page_config(page_title="UK Lawyer Assistant", layout="wide")
st.title("⚖️ UK Lawyer Assistant")
st.caption("Generate legal search links — optionally summarize with Ollama (llama3.1)")

# Sidebar with tips
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use `"exact phrases"`  
    - Combine: `"child custody" AND "High Court"`  
    - Try statute names: `"Human Rights Act 1998"`  
    - Family law uses anonymized names: `X v Y`
    """)

# ----------------------------
# User Input
# ----------------------------
st.markdown("### 🔍 Enter Legal Query")
query = st.text_input(
    "Keywords or legal question",
    placeholder='e.g., "unfair dismissal Employment Tribunal"',
    label_visibility="collapsed"
)

case_type = st.selectbox(
    "Case Type",
    ["Civil", "Criminal", "Family / Marriage", "Other"]
)

search_scope = st.radio(
    "Search Scope",
    ["Internet (Google)", "Specific Legal Site"]
)

selected_site = None
if search_scope == "Specific Legal Site":
    selected_site = st.selectbox(
        "Choose site to search",
        ["BAILII", "Family Court (gov.uk)", "UK Legislation", "Supreme Court", "Court of Appeal / High Court"]
    )

if not query.strip():
    st.info("👆 Enter a legal query to continue.")
    st.stop()

q = urllib.parse.quote_plus(query.strip())

# ----------------------------
# Generate Search Links
# ----------------------------
st.markdown("### 🔗 Search Links")

# Internet / Google
if search_scope == "Internet (Google)":
    google_url = f"https://www.google.com/search?q={q}"
    st.markdown(f"#### 🔍 [Google Search](%s)" % google_url)

# Site-specific wrappers
elif search_scope == "Specific Legal Site":
    if selected_site == "BAILII":
        bailii_url = f"https://www.bailii.org/cgi-bin/sino_search.cgi?query={q}"
        st.markdown(f"#### 📘 [BAILII – UK & Irish Case Law]({bailii_url})")
    elif selected_site == "Family Court (gov.uk)":
        family_url = f"https://www.gov.uk/search/news-and-communications?content_store_document_type=family-court-decisions&keywords={q}"
        st.markdown(f"#### 👨‍👩‍👧 [Family Court Decisions (gov.uk)]({family_url})")
    elif selected_site == "UK Legislation":
        legislation_url = f"https://www.legislation.gov.uk/secondary?text={q}"
        st.markdown(f"#### 📜 [UK Legislation]({legislation_url})")
    elif selected_site == "Supreme Court":
        supreme_url = f"https://www.supremecourt.uk/search-judgments/results.html?query={q}"
        st.markdown(f"#### ⚖️ [Supreme Court Judgments]({supreme_url})")
    elif selected_site == "Court of Appeal / High Court":
        court_url = f"https://www.bailii.org/cgi-bin/sino_search.cgi?query={q}"
        st.markdown(f"#### 🏛️ [Court of Appeal / High Court via BAILII]({court_url})")
    else:
        st.warning("Select a valid legal site.")

# ----------------------------
# Optional Ollama Summarization
# ----------------------------
st.markdown("---")
st.subheader("🧠 Optional: Summarize Legal Context with Ollama")

use_ollama = st.checkbox("✅ Use Ollama (`llama3.1`) to generate a legal summary")

if use_ollama:
    pasted_text = st.text_area(
        "Paste relevant legal text (e.g., judgment excerpt, statute)",
        height=150,
        placeholder="Copy text from BAILII or gov.uk to summarize..."
    )

    if st.button("✨ Generate Summary", type="primary"):
        if not pasted_text.strip():
            st.warning("Please paste legal text to summarize.")
        else:
            try:
                from langchain_ollama import ChatOllama

                prompts = {
                    "Family / Marriage": "You are a UK family law barrister. Summarize the key legal principles, facts, and outcome. Highlight child welfare, financial settlements, or precedent value.",
                    "Criminal": "You are a UK criminal defence solicitor. Extract the charge, key facts, legal arguments, court reasoning, and sentence.",
                    "Civil": "You are a civil litigator. Summarize the claim, defences, applicable law, findings, and remedy awarded.",
                    "Other": "You are a UK-qualified lawyer. Provide a concise, accurate legal summary."
                }
                system_msg = prompts.get(case_type, prompts["Other"])

                with st.spinner("Generating summary with **llama3.1**..."):
                    llm = ChatOllama(
                        model="llama3.1",
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                        temperature=0.1,
                        num_ctx=4096
                    )
                    response = llm.invoke(f"{system_msg}\n\nText:\n{pasted_text.strip()}")

                st.success("✅ Summary generated!")
                st.info(response.content)

                st.download_button(
                    "📥 Download Summary",
                    data=response.content,
                    file_name=f"legal_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

            except ImportError:
                st.error("❌ Install `langchain-ollama`: `pip install langchain-ollama`")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Make sure Ollama is running (`ollama serve`) and `llama3.1` is pulled.")
else:
    st.info("Toggle the checkbox above if you'd like to generate an AI summary from pasted legal text.")
