import streamlit as st
from langchain.llms import GeminiLLM  # Ensure you have the correct Gemini LLM integration installed
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Fetch the Gemini API key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

def generate_journal_report(visit_date, visit_details, questions):
    prompt_template = PromptTemplate(
        input_variables=["date", "visit_details", "questions"],
        template="""
You are a social welfare expert. Based on the following details from today's field visit, please draft a comprehensive journal report of approximately 500 words that reflects on the social welfare impact and field activities. Follow the structure below:

1. Please describe the plan of action for today’s field visit. (Include the date and time, objectives, goals, and the purpose of your visit.)
2. Please describe the activities carried out to complete the action plan. (Outline the work done during the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)

Here are the details:
Date of Visit: {date}
Objectives, Goals, and Purpose: {visit_details}
Field Visit Questions/Details: {questions}
        """
    )
    # Instantiate the Gemini LLM with the API key
    llm = GeminiLLM(api_key=GEMINI_API_KEY)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    # Format the date as needed (YYYY-MM-DD)
    formatted_date = visit_date.strftime("%Y-%m-%d")
    report = chain.run({"date": formatted_date, "visit_details": visit_details, "questions": questions})
    return report

def main():
    st.title("Field Visit Journal Report Generator")
    st.write("Enter the details of your field visit so that the LLM can generate a comprehensive report.")

    # 1. Date input for the field visit
    visit_date = st.date_input("Enter the date of the field visit")

    # 2. Free text input for objectives, goals, and purpose of visit
    visit_details = st.text_area(
        "Enter the objectives, goals, and purpose of your visit:",
        "Enter objectives, goals, and purpose of visit here..."
    )

    # 3. Pre-filled text area for the four field visit questions (editable by the user)
    default_questions = """1. Please describe the plan of action for today’s field visit. (Include the date and time, objectives, goals, and the purpose of your visit.)
2. Please describe the activities carried out to complete the action plan. (Outline the work done during the field visit.)
3. What did you observe today that you would like to implement in your next field visit?
4. What are the key learning outcomes from this field visit? (Highlight the lessons learned from the experience.)
"""
    questions = st.text_area(
        "Edit or add details for the field visit questions:",
        default_questions,
        height=200
    )

    # 4. Generate the journal report on button click
    if st.button("Generate Journal Report"):
        if not visit_details or not questions:
            st.error("Please provide both the visit details and responses for the field visit questions.")
        else:
            with st.spinner("Generating report..."):
                report = generate_journal_report(visit_date, visit_details, questions)
            st.subheader("Draft Journal Report")
            st.write(report)
            
            # 5. Download button for the generated report
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name="journal_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
