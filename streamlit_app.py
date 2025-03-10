#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import io

# Google Generative AI & LangChain
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# PDF extraction
from pdfminer.high_level import extract_text

# Set page config first
st.set_page_config(page_title="Karma Yoga Journal Agent", page_icon="📝", layout="wide")

# Sidebar: library versions (optional)
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")

# Check for API key in Streamlit secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found in secrets. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

# Configure Google Generative AI with your API key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Cache the main Gemini model
@st.cache_resource
def load_main_llm():
    """
    Loads a ChatGoogleGenerativeAI model for generating the final journal report.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # Adjust model name/version if needed
        temperature=0.7,
        max_tokens=5000
    )

# Cache a second instance for summarization (can be the same or a smaller model)
@st.cache_resource
def load_summarizer_llm():
    """
    Loads a ChatGoogleGenerativeAI model for summarizing the PDF content.
    You could use the same model or a smaller one if you have access.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.3,
        max_tokens=4000
    )

main_llm = load_main_llm()
summarizer_llm = load_summarizer_llm()

# Prompt for the final Karma Yoga Journal
journal_prompt = """
You are a social welfare expert. Based on the following details from today's field visit, please draft
a comprehensive journal report of approximately 500 words that reflects on the social welfare impact
and field activities. Dont mention visiting Date in the paragraphs. Follow the structure below:

1. Please describe the plan of action for today’s field visit. (Include the date and time, objectives,
   goals, and the purpose of your visit.)
2. Please describe the activities carried out to complete the action plan. (Outline the work done during
   the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)

Here is a summary of the previous report: 
{previous_report_summary}

Project: {project}
Date of Visit: {visit_date}
Objectives, Goals, and Purpose: {visit_details}
Field Visit Questions/Details: {questions}
User's Actions So Far: {actions}

Please ensure this new journal is a true continuation of the previous report, with minimal overlap.
"""

prompt_template = PromptTemplate(
    input_variables=["previous_report_summary", "project", "visit_date", "visit_details", "questions", "actions"],
    template=journal_prompt
)

journal_chain = LLMChain(prompt=prompt_template, llm=main_llm)

# Summarization chain to handle previous PDF
def summarize_pdf_text(pdf_text: str) -> str:
    """
    Summarize the extracted PDF text using a summarization chain with the summarizer LLM.
    This helps keep the final prompt from exceeding token limits if the PDF is large.
    """
    # Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)

    # Convert each chunk into a Document
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Use the Summarize chain
    summarize_chain = load_summarize_chain(summarizer_llm, chain_type="map_reduce")
    summary = summarize_chain.run(docs)
    return summary.strip()

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file using pdfminer.six.
    """
    pdf_bytes = pdf_file.read()
    return extract_text(io.BytesIO(pdf_bytes))

def generate_journal_report(
    previous_report_summary: str,
    project: str,
    visit_date,
    visit_details: str,
    questions: str,
    actions: str
) -> str:
    """
    Generate the final journal report, incorporating the previous report summary.
    """
    # Convert the date input to a string if needed
    date_str = visit_date.strftime("%Y-%m-%d") if visit_date else "No date provided"

    return journal_chain.run({
        "previous_report_summary": previous_report_summary,
        "project": project,
        "visit_date": date_str,
        "visit_details": visit_details,
        "questions": questions,
        "actions": actions
    })

def main():
    st.title("Karma Yoga Journal Report Generator")
    st.write("Upload your media, optionally upload a PDF of a previous report, and provide inputs for your new report.")

    # 1. File uploader for multiple photos and videos (optional)
    uploaded_files = st.file_uploader(
        "Upload Photos and Videos (optional)",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )

    if uploaded_files:
        st.write("### Uploaded Files")
        for uploaded_file in uploaded_files:
            st.write(uploaded_file.name)
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, width=200)
            elif uploaded_file.type.startswith("video/"):
                st.video(uploaded_file)

    # 2. (Optional) PDF of a previous report
    st.subheader("Previous Report (PDF)")
    previous_pdf = st.file_uploader("Upload a PDF of your previous report (optional)", type=["pdf"])
    previous_report_summary = ""
    if previous_pdf is not None:
        with st.spinner("Extracting text from the PDF..."):
            pdf_text = extract_text_from_pdf(previous_pdf)
        if pdf_text:
            st.success(f"Successfully extracted {len(pdf_text)} characters from previous report.")
            # Summarize the PDF to keep context concise
            with st.spinner("Summarizing previous report..."):
                previous_report_summary = summarize_pdf_text(pdf_text)
            st.info("Previous report summarized. This summary will be used for context.")

    # 3. Dropdown menu for selecting Karma Yoga project
    project = st.selectbox("Select your Karma Yoga Project", ["Tree Plantation Drive", "Anti-Drug & Addiction awareness program", "Beach Cleaning Drive", "Mobile Veterinary Camp", "Health Camp", "Road Safety Awareness Campaign", "Gardening Workshop", "Waste Management Awareness", "Financial Literacy"])

    # 4. Date input for the field visit
    visit_date = st.date_input("Enter the date of the field visit")

    # 5. Free text for objectives, goals, and purpose of visit
    visit_details = st.text_area(
        "Enter the objectives, goals, and purpose of your visit:"
    )

    # 6. Pre-filled text area for the four skeleton questions (editable by the user)
    default_questions = """1. Please describe the plan of action for today’s field visit.
2. Please describe the activities carried out to complete the action plan.
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit?
"""
    questions = st.text_area(
        "Edit or add details for the field visit questions:",
        default_questions,
        height=200
    )

    # 7. Text area for user to describe what they have done so far
    actions = st.text_area("Describe what you have done so far:")

    # Generate the journal report
    if st.button("Generate Journal Report"):
        # Basic validation
        if not visit_details or not questions or not actions:
            st.error("Please fill in all the required fields before generating the report.")
        else:
            with st.spinner("Generating your new journal report..."):
                report = generate_journal_report(
                    previous_report_summary=previous_report_summary,
                    project=project,
                    visit_date=visit_date,
                    visit_details=visit_details,
                    questions=questions,
                    actions=actions
                )
            st.subheader("Draft Journal Report")
            st.write(report)

            # Option to download the report as a text file
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name="journal_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()

# Footer for Credits
st.markdown("""---""")
st.markdown(
    """
    <div style="background: linear-gradient(to right, blue, purple); padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px; color: white;">
        Made with ❤️ by Anubhav Verma<br>
        Please reach out to anubhav.verma360@gmail.com if you encounter any issues.
    </div>
    """, 
    unsafe_allow_html=True
)
