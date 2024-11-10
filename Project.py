import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import io
import altair as alt
import plotly.express as px

from PIL import Image


#######################
# Page configuration
st.set_page_config(
    page_title="Student Performance Factors", # Replace this with your Project's Title
    page_icon="📚", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Tyrell John C. Del Carmen\n2. Claraence Paragoso\n3. Markus Antwone Nadora\n4. Monti Kilayco")

    st.markdown("This Project is about the factors that are affecting the Student's performance in their academics.")
    st.markdown("[Dataset](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors?resource=download)")
    st.markdown("[Google Colab Notebook](https://colab.research.google.com/drive/1IYyDVrqw_oaxK7jrronYDpO3x_gCG_4v?usp=sharing)")
    st.markdown("[Github Repository](https://github.com/Lixir81/CSS145-Project.git)")

    

#######################
# Data

# Load data
url = https://raw.githubusercontent.com/Lixir81/CSS145-Project/refs/heads/main/StudentPerformanceFactors.csv
df=read.csv(url)
average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown(""" 
                
        This is a streamlit application where it analyzes the factors that affect the performance of students.
        ###
        1. Dataset - Introduction of the dataset that is used for this application
        2. EDA - Exploratory Data Analysis of the Student Performance Factors dataset. This includes graphs such as Bar Chart, Scatter Chart, etc.
        3. Data Cleaning - Shows the process of data cleaning.
        4. Machine Learning - Process of training of the Machine.
        5. Prediction - This page predicts the outcome of students performance.
        6. Conclusion - Summary and observations on the dataset.
                
                """)


    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")


    st.write("Student Performance Factors Dataset")
    st.write("")

    st.markdown("""
                This dataset provides a overview of factors that affect the student performance in their exams.

                The graphs showed the following factors that affect the performance of students on exams. Such as peer influence, hours studied, parental involvement, access to resources 
                and more.

                Link: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors?resource=download
                
                """)
    st.write(df.describe())
    
    st.markdown("""
                This is the results from the df.describe(). It presents the mean from each columns. However, much of these datas have high deviation which shows inconsistencies.
                The study hours ranges from 1 hour to 44 hours. Attendance ranges from 60 to 100 percent. Previous scores range from 50 to 100. Tutoring sessions range from 0 to 8. 
                Physical Activity level range from 0 to 6. And the final exam score ranges from 55 to 101.

                """)

    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((5.5, 4.5, 4), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('Average Exam Score by Hours Studied')
        plt.figure(figsize=(12, 12))
        plt.bar(average_scores.index, average_scores.values)
        plt.title("Average Exam Score by Hours Studied")
        plt.xlabel("Hours Studied")
        plt.ylabel("Average Exam Score")
        plt.xticks(average_scores.index)
        plt.grid(axis='y')
        st.pyplot(plt)
        plt.clf()


    with col[1]:
        st.markdown('Peer Influence vs Exam Score')
        plt.figure(figsize=(6, 6))
        plt.scatter(df['Peer_Influence'], df['Exam_Score'])
        plt.title("Peer Influence vs Exam Score")
        plt.xlabel("Peer Influence")
        plt.ylabel("Exam Score")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
        
    with col[2]:
        st.markdown('Average Exam Score by Attendance Range')

        attendance_bins = [0, 20, 40, 60, 80, 100]
        attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        df['Attendance_Range'] = pd.cut(df['Attendance'], bins=attendance_bins, labels=attendance_labels)
        average_score = df.groupby('Attendance_Range')['Exam_Score'].mean()

        plt.figure(figsize=(6, 6))
        plt.bar(average_score.index, average_score.values)
        plt.title("Average Exam Score by Attendance Range")
        plt.xlabel("Attendance Range")
        plt.ylabel("Average Exam Score")
        plt.grid(axis='y')
        st.pyplot(plt)
        plt.clf()


    
    st.header("💡Discussion")
    st.subheader("Average Exam Score by Hours Studied")
    st.write(""" This graph shows the average scores of the students depending on how many hours they studied for the exam. The X-axis is the hours presented while the Y-axis is the Average Exam Score.
It is visually presented that the more hours students put into their studies the higher the score goes. The graph has an uptrend position. There may be hours that have lower results than their hour before. But the main path of the graph is that the higher the hours, the higher the score. """)
    
    st.subheader("Peer Influence vs  Exam Score")
    st.write(""" The Graph shows that Positive Peer Influence has proven to achieve greater scores for the students. There are a lot of points where students scored around 80-90 on the positive. 
             While Negative have consisdently scored below 80. Neutral had better results than Negative but is still subpar to the positive peer influence. """)
    
    st.subheader("Average Exam Score by Attendance Range")
    st.write(""" This graph shows the relationship of attendance and the exam score. It is visually shown that the higher the attendance range for the student, the better the exam scores
             they usually get. The graph is in an uptrend position where the higher one variable is, the higher the other one also is. Which means that more scores usually means more attendance
             and vice versa.  """)
    

             



# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write(""" We use df.info in ordet to look for null values""")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
