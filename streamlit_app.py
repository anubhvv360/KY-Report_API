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

###############################################################################
# Prompt template for a Social Impact based journal report

journal_prompt = """
You are a social welfare expert. Based on the following details from today's field visit, please draft a comprehensive jou
