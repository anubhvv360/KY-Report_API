#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import google.generativeai as genai
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the Streamlit page
st.set_page_config(page_title="Karma Yoga Journal", page_icon="📝", layout="wide")

# Sidebar: show library versions (optional)
st.sidebar.markdown("### Library Versions")
st.sidebar.markdown(f"google.generativeai: {genai.__version__}")
st.sidebar.markdown(f"streamlit: {st.__version__}")
st.sidebar.markdown(f"langchain: {langchain.__version__}")

# Check for API key in Streamlit secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found in secrets. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

# Configure Google Generative AI with your API key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Cache the LLM resource so it isn't reloaded on every run
@st.cache_resource
def load_llm():
    """
    Loads a ChatGoogleGenerativeAI model from the langchain_google_genai package,
    configured for the Gemini model. Adjust the parameters as needed.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=5000
    )

llm = load_llm()

# Define the prompt template
prompt_template = """
You are a social welfare expert. Based on the following details from today's field visit, please draft
a comprehensive journal report of approximately 500 words that reflects on the social welfare impact
and field activities. Follow the structure below:

1. Please describe the plan of action for today’s field visit. (Include the date and time, objectives,
   goals, and the purpose of your visit.)
2. Please describe the activities carried out to complete the action plan. (Outline the work done during
   the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)

Project: {project}
Date of Visit: {visit_date}
Objectives, Goals, and Purpose: {visit_details}
Field Visit Questions/Details: {questions}
User's Actions So Far: {actions}

Include relevant social welfare reflections and ensure the tone is both formal and empathetic.
"""

template = PromptTemplate(
    input_variables=["project", "visit_date", "visit_details", "questions", "actions"],
    template=prompt_template
)

# Create an LLMChain
llm_chain = LLMChain(prompt=template, llm=llm)

# Function to generate the journal report
def generate_journal_report(project, visit_date, visit_details, questions, actions):
    # Convert the date input to a string if needed
    date_str = visit_date.strftime("%Y-%m-%d") if visit_date else "No date provided"
    
    return llm_chain.run({
        "project": project,
        "visit_date": date_str,
        "visit_details": visit_details,
        "questions": questions,
        "actions": actions
    })

def main():
    st.title("Karma Yoga Journal Report Generator")
    st.write("Upload your media, select your project, and provide your inputs so the LLM can generate your report.")

    # 1. File uploader for multiple photos and videos
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

    # 2. Dropdown menu for selecting Karma Yoga project
    project = st.selectbox("Select your Karma Yoga Project", ["project 1", "project 2"])

    # 3. Date input for the field visit
    visit_date = st.date_input("Enter the date of the field visit")

    # 4. Free text for objectives, goals, and purpose of visit
    visit_details = st.text_area(
        "Enter the objectives, goals, and purpose of your visit:",
        "Enter objectives, goals, and purpose of visit here..."
    )

    # 5. Pre-filled text area for the four skeleton questions (editable by the user)
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

    # 6. Text area for user to describe what they have done so far
    actions = st.text_area("Describe what you have done so far:")

    # Generate the journal report
    if st.button("Generate Journal Report"):
        if not actions or not questions or not visit_details:
            st.error("Please fill in all the required fields before generating the report.")
        else:
            with st.spinner("Generating report..."):
                report = generate_journal_report(project, visit_date, visit_details, questions, actions)
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
