#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import io
import time

# Google Generative AI & LangChain imports
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# PDF extraction
from pdfminer.high_level import extract_text

# Set page config
st.set_page_config(page_title="Karma Yoga Journal Agent", page_icon="📝", layout="wide")

# Sidebar: library versions (optional)
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")

###############################################################################
# Step 1: Ask the user for their Google API Key
if "api_key_entered" not in st.session_state:
    st.title("Karma Yoga Journal Report Generator")
    with st.form("api_key_form"):
        st.header("Enter Your GOOGLE_API_KEY")
        st.info(
            "You need a Google API Key to use Google's Generative AI.\n\n"
            "If you don't have one, you can create it here:\n"
            "[Create a Google API Key](https://console.cloud.google.com/apis/credentials)"
        )
        user_api_key = st.text_input("Enter your Google API Key", type="password")
        submitted = st.form_submit_button("Next")
        if submitted:
            if user_api_key.strip():
                genai.configure(api_key=user_api_key.strip())
                st.session_state.api_key_entered = True
                st.success("API Key configured successfully! Please proceed below.")
                st.stop()
            else:
                st.error("API Key is required to proceed.")
    st.stop()

###############################################################################
# Prompt template for the journal report

journal_prompt = """
You are a social welfare expert. Based on the following details from today's field visit, please draft 
a comprehensive journal report of approximately 500 words that reflects on the social welfare impact 
and field activities. Don't mention the visiting date in the paragraphs. Follow the structure below:

1. Please describe the plan of action for today’s field visit. (Include objectives, goals, and the purpose of your visit.)
2. Please describe the activities carried out to complete the action plan. (Outline the work done during the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)

Here is a summary of the previous report:
{previous_report_summary}

Project: {project}
Date of Visit: {visit_date}
Visit Number: {visit_number}
User's Actions So Far: {actions}

Please ensure this new journal is a true continuation of the previous report, with minimal overlap.
"""

prompt_template = PromptTemplate(
    input_variables=["previous_report_summary", "project", "visit_date", "visit_number", "actions"],
    template=journal_prompt
)

journal_chain = LLMChain(
    prompt=prompt_template,
    llm=ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # Adjust model name/version if needed
        temperature=0.7,
        max_tokens=5000
    )
)

###############################################################################
# Cached summarization to avoid repeated runs if the same text is provided
@st.cache_data(show_spinner=False)
def cached_summarize_pdf_text(pdf_text: str) -> str:
    """
    Summarize the extracted PDF text using a summarization chain with the summarizer LLM.
    The result is cached to avoid re-running the summarization unnecessarily.
    """
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    summarize_chain = load_summarize_chain(
        ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-8b",
            temperature=0.3,
            max_tokens=4000
        ),
        chain_type="map_reduce"
    )
    summary = summarize_chain.run(docs)
    return summary.strip()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from an uploaded PDF file using pdfminer.six."""
    pdf_bytes = pdf_file.read()
    return extract_text(io.BytesIO(pdf_bytes))

def generate_journal_report(
    previous_report_summary: str,
    project: str,
    visit_date,
    visit_number: str,
    actions: str
) -> str:
    """Generate the final journal report, incorporating the previous report summary."""
    date_str = visit_date.strftime("%Y-%m-%d") if visit_date else "No date provided"
    return journal_chain.run({
        "previous_report_summary": previous_report_summary,
        "project": project,
        "visit_date": date_str,
        "visit_number": visit_number,
        "actions": actions
    })

###############################################################################
# Main app logic

def main():
    st.title("Karma Yoga Journal Report Generator")
    st.write("Provide the details for your field visit. You may optionally upload a PDF of a previous report (for context).")

    # 1. General Information Section
    st.subheader("General Information")
    project = st.selectbox("Select your Karma Yoga Project", [
        "Tree Plantation Drive",
        "Anti-Drug & Addiction Awareness Program",
        "Beach Cleaning Drive",
        "Mobile Veterinary Camp",
        "Health Camp",
        "Road Safety Awareness Campaign",
        "Gardening Workshop",
        "Waste Management Awareness",
        "Financial Literacy"
    ])
    visit_number = st.selectbox("Which visit is it?", [
        "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"
    ])
    visit_date = st.date_input("Enter the date of the field visit")
    actions = st.text_area("Describe what you have done so far:")

    # 2. Optional: Upload previous report PDF
    st.subheader("Previous Report (PDF) - Optional")
    previous_pdf = st.file_uploader("Upload your previous report (PDF only)", type=["pdf"])
    previous_report_summary = ""
    if previous_pdf is not None:
        with st.spinner("Extracting text from the PDF..."):
            pdf_text = extract_text_from_pdf(previous_pdf)
        if pdf_text:
            st.success(f"Successfully extracted {len(pdf_text)} characters from the previous report.")
            with st.spinner("Summarizing previous report..."):
                previous_report_summary = cached_summarize_pdf_text(pdf_text)
            st.info("Previous report summarized. This summary will be used for context building.")

    # 3. Generate the journal report
    if st.button("Generate Journal Report"):
        if not actions:
            st.error("Please describe what you have done so far.")
        else:
            with st.spinner("Generating your new journal report..."):
                report = generate_journal_report(
                    previous_report_summary=previous_report_summary,
                    project=project,
                    visit_date=visit_date,
                    visit_number=visit_number,
                    actions=actions
                )
            st.subheader("Draft Journal Report")
            st.write(report)
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name="journal_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    # If the user provided an API key, proceed with main; otherwise we won't proceed
    if st.session_state.get("api_key_entered"):
        main()
    else:
        st.stop()

# Footer for Credits
st.markdown("---")
st.markdown(
    """
    <div style="background: linear-gradient(to right, blue, purple); padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; color: white;">
        Made with ❤️ by Anubhav Verma<br>
        Please reach out to anubhav.verma360@gmail.com if you encounter any issues.
    </div>
    """,
    unsafe_allow_html=True
)
