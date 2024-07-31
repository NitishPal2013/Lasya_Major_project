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

class Question(BaseModel):
    question: str = Field(description="The question")
    options: list[str] = Field(description="list of 4 options for the question with one correct and other incorrect")
    correct: str = Field(description="correct option or answer of the question")

class QuestionList(BaseModel):
    case_study: str = Field(description="generated case study for the questions")
    questions: list[Question] = Field(description="list of 5 question, options and correct option")


class CaseStudyQuestions(BaseModel):
    case_studies: list[QuestionList] = Field(description="list of 5 case studies and its questions.")


API_KEY = os.environ["GOOGLE_API_KEY"]
gemini_model = GoogleGenerativeAI(model="gemini-1.5-flash-001",temperature=0.7, google_api_key=API_KEY)
parser = JsonOutputParser(pydantic_object=CaseStudyQuestions)


prompt = PromptTemplate(template="""
### Objective

You are an AI designed to generate case study-based multiple-choice questions (MCQs) from the given context.

### Steps

1. **Comprehend the Content**

   - Carefully read and understand the provided context.
   - Identify key concepts, important details, and overall context.

2. **Generate Case Study-Based MCQs**

   - Create a scenario or case study based on the content.
   - Ensure the case study is relevant and logically derived from the given material.

3. **Formulate Questions**

   - Develop multiple-choice questions (MCQs) based on the case study.
   - Each question should have one correct answer and three plausible distractors (incorrect options).

4. **Maintain Quality and Difficulty**

   - Ensure questions vary in difficulty, including some that test basic understanding and others that challenge deeper comprehension.
   - Questions should test application, analysis, and evaluation based on the case study.

### Output Format

Generate the scenario, questions, and options in JSON format with the following structure:

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

    # print(questions)

    # Display questions
    st.write("**Questions:**")
    if questions:
        for i, case in enumerate(questions["case_studies"]):
            st.write(f"**Case {i+1} : {case['case_study']}**")
            for j, question in enumerate(case["questions"]):
                st.write(f"**Q{j+1}: {question['question']}**")
                st.radio("Options : ",question['options'], key= f"{j+1 + 5*i}")
        
        st.write("Correct Options")
        for i, case in enumerate(questions["case_studies"]):
            st.write(f"**Case {i+1} : {case['case_study']}**")
            for j, question in enumerate(case["questions"]):
                st.write(f"**Q{j+1}: {question['question']}**")
                st.write(f"Answer: {question['correct']}")

        # Submit button
        # if st.button("Submit"):
        #     score = 0
        #     correct_answers = []
        #     wrong_answers = []
        #     for i, question in enumerate(questions["questions List"]):
        #         selected_option = st.session_state.get(f"{i}")
        #         if selected_option == question['correct']:
        #             score += 1
        #             correct_answers.append(question['question'])
        #         else:
        #             wrong_answers.append(question['question'])

        #     # Display score and answers
        #     st.write("**Score:**", score)
        #     st.write("**Correct Answers:**", correct_answers)
        #     st.write("**Wrong Answers:**", wrong_answers)
    else:
        st.write("No questions generated.")