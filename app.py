import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import json
import os
from langchain.globals import set_verbose

# Set the verbosity level to True to enable verbose logging
set_verbose(True)
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

class question(BaseModel):
    question: str = Field(description="The question")
    options: list[str] = Field(description="list of 4 options for the question")
    correct: str = Field(description="correct option or answer of the question")


class QuestionList(BaseModel):
    questions: list[question] = Field(alias="questions List")

API_KEY = os.environ["GOOGLE_API_KEY"]
gemini_model = GoogleGenerativeAI(model="gemini-1.5-flash-001",temperature=0.5, google_api_key=API_KEY)
parser = JsonOutputParser(pydantic_object=QuestionList)

prompt = PromptTemplate(template="""
Task: Generate 15 multiple choice questions and their answers from within the given context.
      Some of them should be straight forward for knowledge test and some of them should be like **Case studies** or **Scenario Based**.
Remeber:
    - Do not provide the same questions present in context.
    - You need to generate questions like present in context with some more difficulty and technicality.
    - Provide the questions in such manner so that user can test his/her Knowledge completely.
Output format:
{format_instructions}                        

Context: 
{context}
""", input_variables=["context"],partial_variables={"format_instructions": parser.get_format_instructions()})

chain = {"context": RunnablePassthrough()} | prompt | gemini_model | parser

def generate_questions(content : str):
    try:
        res = chain.invoke(content)
        return res
    except Exception as e:
        # Log the error internally and continue
        print(f"Error generating questions: {e}")
        return None

def format_docs(docs):
    return "/n".join(doc.page_content for doc in docs)


# loader = PyMuPDFLoader("./uploaded_pdf.pdf")
# docs = loader.load()
# content = format_docs(docs)

# print(content)


st.title("Multiple Choice Question Generator")
st.markdown("Upload a PDF file and we'll generate multiple choice questions and answers for you.")

uploaded_file = st.file_uploader("Choose a PDF file")

if uploaded_file is not None:
    # Save the uploaded file locally
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the uploaded PDF file
    loader = PyMuPDFLoader("uploaded_pdf.pdf")
    docs = loader.load()
    content = format_docs(docs)
    # print(content)
    try:
        questions = generate_questions(content)        
    except Exception as e:
        # Log the error internally and continue
        print(f"Error in question generation loop: {e}")

    # Display questions
    st.write("**Questions:**")
    if questions:
        for i, question in enumerate(questions["questions List"]):
            st.write(f"**Q: {question['question']}**")
            st.radio("Options:", question["options"], key=f"{i}")
        
        # Submit button
        if st.button("Submit"):
            score = 0
            correct_answers = []
            wrong_answers = []
            for i, question in enumerate(questions["questions List"]):
                selected_option = st.session_state.get(f"{i}")
                if selected_option == question['correct']:
                    score += 1
                    correct_answers.append(question['question'])
                else:
                    wrong_answers.append(question['question'])

            # Display score and answers
            st.write("**Score:**", score)
            st.write("**Correct Answers:**", correct_answers)
            st.write("**Wrong Answers:**", wrong_answers)
    else:
        st.write("No questions generated.")