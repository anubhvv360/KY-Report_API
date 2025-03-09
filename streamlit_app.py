#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import io

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

###############################################################################
# Helper functions

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from an uploaded PDF file using pdfminer.six."""
    pdf_bytes = pdf_file.read()
    return extract_text(io.BytesIO(pdf_bytes))

def summarize_pdf_text(pdf_text: str) -> str:
    """
    Summarize the extracted PDF text using a summarization chain with the summarizer LLM.
    This keeps the final prompt concise.
    """
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    summarize_chain = load_summarize_chain(summarizer_llm, chain_type="map_reduce")
    summary = summarize_chain.run(docs)
    return summary.strip()

###############################################################################
# Prompt template for a Social Impact based journal report

prompt_text = """
You are a world-class social welfare expert with years of experience evaluating field visits. Based on the context provided below, please draft a detailed journal report (500 to 700 words) that convincingly reflects the social impact of the current field visit. The report must be specific and persuasive to a seasoned evaluator. Don't mention any date in the report. Make the output in plural first person (using We/Us)

Structure:
1. Describe the plan of action for today’s field visit. (Include objectives, goals, and the purpose of your visit.)
2. Describe the activities carried out to complete the action plan. (Outline the work done during the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)

Context:
- Previous Visit Report Summary: {previous_report_summary}
- Karma Yoga Project: {project}
- Visit Number: {visit_number}
- Current Visit Details: {current_visit_details}
- Current Visit Media Summary: {media_summary}
- Visit Date: {visit_date}

Ensure that your report builds on the previous context, includes details from uploaded media (if any), and is tailored to exhibit strong social impact.
"""

prompt_template = PromptTemplate(
    input_variables=["previous_report_summary", "project", "visit_number", "current_visit_details", "media_summary", "visit_date"],
    template=prompt_text
)

###############################################################################
# Load LLM models (using caching)

@st.cache_resource
def load_main_llm():
    """Load the main ChatGoogleGenerativeAI model for generating the journal report."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # Adjust as needed
        temperature=0.7,
        max_tokens=8000
    )

@st.cache_resource
def load_summarizer_llm():
    """Load a ChatGoogleGenerativeAI model for summarizing previous PDF reports."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=8000
    )

main_llm = load_main_llm()
summarizer_llm = load_summarizer_llm()
journal_chain = LLMChain(prompt=prompt_template, llm=main_llm)

def generate_journal_report(previous_report_summary: str, project: str, visit_number: str,
                            current_visit_details: str, media_summary: str, visit_date) -> str:
    """Generate the final journal report using the provided context."""
    date_str = visit_date.strftime("%Y-%m-%d") if visit_date else "No date provided"
    return journal_chain.run({
        "previous_report_summary": previous_report_summary,
        "project": project,
        "visit_number": visit_number,
        "current_visit_details": current_visit_details,
        "media_summary": media_summary,
        "visit_date": date_str
    })

###############################################################################
# Multi‑step logic using session_state

if "step" not in st.session_state:
    st.session_state.step = 1

# STEP 1: API Key input
if st.session_state.step == 1:
    with st.form("api_key_form"):
        st.header("Step 1: Enter Your Google API Key")
        st.info("To use this app, you need a Google API Key for Generative AI. "
                "Get one from the [Google AI Studio - Generate API Key](https://aistudio.google.com/apikey).")
        api_key = st.text_input("Google API Key", type="password")
        submitted = st.form_submit_button(label="Next")
        st.caption("Note: Please press 'Next' twice.")
        if submitted:
            if not api_key:
                st.error("API Key is required to proceed.")
            else:
                st.session_state.api_key = api_key
                genai.configure(api_key=api_key)
                st.session_state.step = 2
                st.stop()

# STEP 2: General Information
if st.session_state.step == 2:
    with st.form("general_info_form"):
        st.header("Step 2: General Information")
        project = st.selectbox("Select your Karma Yoga Project", options=[
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
        visit_number = st.selectbox("Which visit is it?", options=["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])
        submitted = st.form_submit_button(label="Next")
        st.caption("Note: Please press 'Next' twice.")
        if submitted:
            st.session_state.project = project
            st.session_state.visit_number = visit_number
            if visit_number == "1st":
                st.session_state.previous_report_summary = ""  # No previous context
                st.session_state.step = 4  # Skip directly to step 4
            else:
                st.session_state.step = 3
            st.stop()

# STEP 3: Previous Visit Information
if st.session_state.step == 3:
    with st.form("previous_visit_form"):
        st.header("Step 3: Previous Visit Information")
        st.info("Upload a PDF report of your previous visit(s) to provide context. (PDF only)")
        previous_pdf = st.file_uploader("Upload Previous Visit Report (PDF)", type=["pdf"])
        submitted = st.form_submit_button("Next")
        st.caption("Note: Please press 'Next' again once the spinner disappears.")
        if submitted:
            if previous_pdf is not None:
                with st.spinner("Extracting and summarizing previous report..."):
                    pdf_text = extract_text_from_pdf(previous_pdf)
                    if pdf_text:
                        summary = summarize_pdf_text(pdf_text)
                        st.session_state.previous_report_summary = summary
                    else:
                        st.session_state.previous_report_summary = ""
            else:
                st.session_state.previous_report_summary = ""
            st.session_state.step = 4
            st.stop()

# STEP 4: Current Visit Information
if st.session_state.step == 4:
    with st.form("current_visit_form"):
        st.header("Step 4: Current Visit Information")
        st.info("Upload any images or videos that show what is being done during this visit (optional).")
        media_files = st.file_uploader("Upload Images/Videos (optional)", 
                                       type=["png", "jpg", "jpeg", "mp4", "mov", "avi"], 
                                       accept_multiple_files=True)
        current_visit_details = st.text_area(
            "Describe what you have done in the current visit:",
            help="Provide clear and detailed information about the work you performed during this visit. Do not include sensitive information."
        )
        visit_date = st.date_input("Enter the visit date")
        submitted = st.form_submit_button(label="Generate Journal Report")
        st.caption("Note: Please press 'Generate Journal Report' again once the spinner disappears to view the report.")
        if submitted:
            st.session_state.media_files = media_files
            st.session_state.current_visit_details = current_visit_details
            st.session_state.visit_date = visit_date

            # Create a media summary from the uploaded files
            if media_files:
                file_names = [f.name for f in media_files]
                media_summary = f"Uploaded media files: {', '.join(file_names)}."
            else:
                media_summary = "No media files provided."

            st.session_state.media_summary = media_summary
            # Generate the final report
            with st.spinner("Generating your journal report..."):
                report = generate_journal_report(
                    previous_report_summary=st.session_state.get("previous_report_summary", ""),
                    project=st.session_state.project,
                    visit_number=st.session_state.visit_number,
                    current_visit_details=st.session_state.current_visit_details,
                    media_summary=st.session_state.media_summary,
                    visit_date=st.session_state.visit_date
                )
            st.session_state.report = report
            st.session_state.step = 5
            st.stop()

# STEP 5: Display Report & Download Option
if st.session_state.step == 5:
    st.header("Journal Report")
    st.write(st.session_state.report)
    st.download_button(
        label="Download Report as Text",
        data=st.session_state.report,
        file_name="journal_report.txt",
        mime="text/plain"
    )

    # Footer: Displayed only in the final step
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
