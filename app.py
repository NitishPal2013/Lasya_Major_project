import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from dotenv import load_dotenv
import os
from langchain.globals import set_verbose

# Set the verbosity level to True to enable verbose logging
set_verbose(True)
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

class Question(BaseModel):
    question: str
    options: list[str]
    answer: str

API_KEY = os.environ["GOOGLE_API_KEY"]
gemini_model = GoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)
parser = JsonOutputParser(pydantic_object=Question)

prompt = PromptTemplate(template="""
Task: Generate at least 2 or a maximum of 10 multiple choice questions and their answers from within the given context.
Output format: The output should be in JSON format.
Here is the field format:
    question: str
    options: list[str]
    answer: str

1) Question 1
    a) option 1
    b) option 2
    c) option 3
    d) option 4

    correct: a) option 1

Given context: {context}
""", input_variables=["context"])

chain = prompt | gemini_model

def generate_questions(doc):
    try:
        res = chain.invoke({"context": doc.page_content})
        res = parser.parse(res)
        return res
    except Exception as e:
        # Log the error internally and continue
        print(f"Error generating questions: {e}")
        return None

st.title("Multiple Choice Question Generator")
st.markdown("Upload a PDF file and we'll generate multiple choice questions and answers for you.")

uploaded_file = st.file_uploader("Choose a PDF file")

if uploaded_file is not None:
    # Save the uploaded file locally
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the uploaded PDF file
    loader = PyPDFLoader("uploaded_pdf.pdf")
    docs = loader.load()

    

    # Generate questions and answers
    questions = []
    try:
        for doc in docs[:2]:
            question = generate_questions(doc)
            if question:
                questions.append(question)
    except Exception as e:
        # Log the error internally and continue
        print(f"Error in question generation loop: {e}")


    # Display questions
    st.write("**Questions:**")
    for i,question in enumerate(questions):
        for j,ques in enumerate(question["questions"]):
            st.write(f"**Q: {ques['question']}**")
            st.radio("Options:",ques["options"], key=f"{i}{j}")

    # Submit button
    if st.button("Submit"):
        score = 0
        correct_answers = []
        wrong_answers = []
        for i,question in enumerate(questions):        
            for j,ques in enumerate(question["questions"]):
                selected_option = st.session_state.get(f"{i}{j}")
                if selected_option == ques['answer']:
                    score += 1
                    correct_answers.append(ques['question'])
                else:
                    wrong_answers.append(ques['question'])

        # Display score and answers
        st.write("**Score:**", score)
        st.write("**Correct Answers:**", correct_answers)
        st.write("**Wrong Answers:**", wrong_answers)
