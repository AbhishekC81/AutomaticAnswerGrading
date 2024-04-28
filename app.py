import streamlit as st
import pandas as pd
from model import run_model

# Streamlit App Title
st.title("Automatic Answer Grading System")

# Sample Answer Input
sample_answer = st.text_area("Enter the sample answer:")

# CSV File Upload
uploaded_file = st.file_uploader("Upload the answer dataset (CSV file):", type=["csv"])

# Total Marks Input
total_marks = st.number_input("Enter the total marks for the question:", value=100)

if sample_answer and uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Call the run_model function
    graded_df = run_model(sample_answer, df, total_marks)

    # Display each answer with final score
    st.write("Graded Answers:")
    st.dataframe(graded_df)
