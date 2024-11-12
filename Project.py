import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from io import StringIO

from PIL import Image

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder


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
url = 'https://raw.githubusercontent.com/Lixir81/CSS145-Project/refs/heads/main/StudentPerformanceFactors.csv'
df = pd.read_csv(url)


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
    col_study = st.columns((3, 3, 3), gap='medium')
    resize_dimensions = (500, 300)

    with col_study[0]:
        st.image("library.jpg")
    with col_study[1]:
        st.image("Studying.jpg")
    with col_study[2]:
        st.image("study.jpg")



    st.subheader("df.describe")
    st.write(df.describe())
    
    st.markdown("""
                This is the results from the df.describe(). It presents the mean from each columns. However, much of these datas have high deviation which shows inconsistencies.
                The study hours ranges from 1 hour to 44 hours. Attendance ranges from 60 to 100 percent. Previous scores range from 50 to 100. Tutoring sessions range from 0 to 8. 
                Physical Activity level range from 0 to 6. And the final exam score ranges from 55 to 101.

                """)

    st.subheader("df")
    df
    st.markdown("""
                This is the complete dataset that will be used for this data analyzation.

                """)
    st.subheader("df.isna().sum()")
    st.write(df.isna().sum())
    st.caption("Missing/Null values by df.isna().sum()")

    st.subheader("df.columns")
    st.write(df.columns)

    st.subheader("df.nunique()")
    st.write(df.nunique())
    

    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((5.5, 4.5, 4), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()
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

    st.subheader("Linear Regression Hours Studied vs Exam Score.")
    a = df[['Hours_Studied']]
    b = df['Exam_Score']
    
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(a_train, b_train)
    b_pred = model.predict(a_test)
    mse = mean_squared_error(b_test, b_pred)
    r2 = r2_score(b_test, b_pred)
    plt.figure(figsize=(15, 5))
    plt.scatter(a_test, b_pred, color='green', marker='x', label='Test predictions')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.title('Linear Regression: Hours Studied vs Exam Score')
    plt.legend()
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

    buffer = StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.text("DataFrame Info:")
    st.text(info)


    st.write(""" We use df.info in ordet to look for null values""")

    df
    st.write("This is the dataset.")
    st.write("We will be using Linear Regression Model in order to predict the relationship between the attendance of students and their performance in exams and how many hours studied and their exam scores")
    
    attendance_bins = [0, 20, 40, 60, 80, 100]
    attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['Attendance_Range'] = pd.cut(df['Attendance'], bins=attendance_bins, labels=attendance_labels)
    average_score = df.groupby('Attendance_Range')['Exam_Score'].mean()


    plt.figure(figsize=(15, 5))
    plt.bar(average_score.index, average_score.values)
    plt.title("Average Exam Score by Attendance Range")
    plt.xlabel("Attendance Range")
    plt.ylabel("Average Exam Score")
    plt.grid(axis='y')
    st.pyplot(plt)
    plt.clf()
            

    st.subheader("Linear Regression for Attendance and their Score")
    attendance_encoded = pd.get_dummies(df['Attendance_Range'], prefix='Attendance') 
    df = pd.concat([df, attendance_encoded], axis=1)
    X = df[['Attendance_0-20', 'Attendance_21-40', 'Attendance_41-60', 'Attendance_61-80', 'Attendance_81-100']]
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.code("""     attendance_encoded = pd.get_dummies(df['Attendance_Range'], prefix='Attendance') 
    df = pd.concat([df, attendance_encoded], axis=1)
    X = df[['Attendance_0-20', 'Attendance_21-40', 'Attendance_41-60', 'Attendance_61-80', 'Attendance_81-100']]
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) """)

    st.subheader("X_Train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_Test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_Train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_Test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)
    
    
    average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()
    st.markdown('Average Exam Score by Hours Studied')
    plt.figure(figsize=(12, 6))
    plt.bar(average_scores.index, average_scores.values)
    plt.title("Average Exam Score by Hours Studied")
    plt.xlabel("Hours Studied")
    plt.ylabel("Average Exam Score")
    plt.xticks(average_scores.index)
    plt.grid(axis='y')
    st.pyplot(plt)
    plt.clf()
    st.write("The Data that will be used for Hours Studied and Exam Score")

    st.subheader("Linear Regression for Hours Studied and Exam Score")

    a = df[['Hours_Studied']] 
    b = df['Exam_Score']     
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

    st.code("""    a = df[['Hours_Studied']] 
    b = df['Exam_Score']     
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)""")
    
    st.subheader("a_Train")
    st.dataframe(a_train, use_container_width=True, hide_index=True)

    st.subheader("a_Test")
    st.dataframe(a_test, use_container_width=True, hide_index=True)

    st.subheader("b_Train")
    st.dataframe(b_train, use_container_width=True, hide_index=True)

    st.subheader("b_Test")
    st.dataframe(b_test, use_container_width=True, hide_index=True)







    
    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Linear Regression")
    
    
    st.markdown(""" Linear Regression is a model where it is used to predict a value of one variable by basing on the another variable. As the name implies, linear regression
                has a straight line that minimizes the errors on the predicted and the actual values of the datas.

                """)
    st.image('LinearRegressionSample.jpg', caption = 'An example of Linear Regression.')

    st.subheader("Creating and Training the Model")
    attendance_bins = [0, 20, 40, 60, 80, 100]
    attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['Attendance_Range'] = pd.cut(df['Attendance'], bins=attendance_bins, labels=attendance_labels)
    average_score = df.groupby('Attendance_Range')['Exam_Score'].mean()
    plt.figure(figsize=(15, 5))
    plt.bar(average_score.index, average_score.values)
    plt.title("Average Exam Score by Attendance Range")
    plt.xlabel("Attendance Range")
    plt.ylabel("Average Exam Score")
    plt.grid(axis='y')
    st.pyplot(plt)
    plt.clf()
    st.caption("Graph that will be used for machine learning")

    st.subheader("Training the Linear Regression Model")
    attendance_encoded = pd.get_dummies(df['Attendance_Range'], prefix='Attendance') 
    df = pd.concat([df, attendance_encoded], axis=1)
    X = df[['Attendance_0-20', 'Attendance_21-40', 'Attendance_41-60', 'Attendance_61-80', 'Attendance_81-100']]
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    st.code ("""     model = LinearRegression()
    model.fit(X_train, y_train)""")

    st.subheader("Test the Accuracy of Data")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.code ("""    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)""")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    viz_df = pd.DataFrame({'Attendance_Range': X_test.idxmax(axis=1).str.replace('Attendance_', ''), 
                       'Actual_Score': y_test, 
                       'Predicted_Score': y_pred})


    viz_df = viz_df.groupby('Attendance_Range')[['Actual_Score', 'Predicted_Score']].mean().reset_index()

    
    average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()
    st.subheader("Second Model of Linear Regression for Average Exam Score")
    st.markdown('Average Exam Score by Hours Studied')
    plt.figure(figsize=(12, 6))
    plt.bar(average_scores.index, average_scores.values)
    plt.title("Average Exam Score by Hours Studied")
    plt.xlabel("Hours Studied")
    plt.ylabel("Average Exam Score")
    plt.xticks(average_scores.index)
    plt.grid(axis='y')
    st.pyplot(plt)
    plt.clf()
    st.write("The Data that will be used for Hours Studied and Exam Score")

    a = df[['Hours_Studied']] 
    b = df['Exam_Score']     
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(a_train, b_train)
      
    st.subheader("Training the Model")
    st.code ("""    model = LinearRegression()
    model.fit(a_train, b_train)""")

    st.subheader("Test the Accuracy of Data")
    b_pred = model.predict(a_test)
    mse = mean_squared_error(b_test, b_pred)
    r2 = r2_score(b_test, b_pred)
    st.code("""    b_pred = model.predict(a_test)
    mse = mean_squared_error(b_test, b_pred)
    r2 = r2_score(b_test, b_pred)""")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")




    # Your content for the MACHINE LEARNING page goes here



# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    
    attendance_bins = [0, 20, 40, 60, 80, 100]
    attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['Attendance_Range'] = pd.cut(df['Attendance'], bins=attendance_bins, labels=attendance_labels)
    average_score = df.groupby('Attendance_Range')['Exam_Score'].mean()
    plt.figure(figsize=(15, 5))
    plt.bar(average_score.index, average_score.values)
    plt.title("Average Exam Score by Attendance Range")
    plt.xlabel("Attendance Range")
    plt.ylabel("Average Exam Score")
    plt.grid(axis='y')

    attendance_encoded = pd.get_dummies(df['Attendance_Range'], prefix='Attendance') 
    df = pd.concat([df, attendance_encoded], axis=1)
    X = df[['Attendance_0-20', 'Attendance_21-40', 'Attendance_41-60', 'Attendance_61-80', 'Attendance_81-100']]
    y = df['Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()
    plt.figure(figsize=(12, 6))
    plt.bar(average_scores.index, average_scores.values)
    plt.title("Average Exam Score by Hours Studied")
    plt.xlabel("Hours Studied")
    plt.ylabel("Average Exam Score")
    plt.xticks(average_scores.index)
    plt.grid(axis='y')





    col = st.columns((5.5, 4.5, 4), gap='medium')
    with col[0]:
        show_dataset = st.checkbox('Show Dataset')
        show_EDA = st.checkbox('Show EDA')
        show_columns = st.checkbox("Show Columns")
        clear_results = st.button('Clear Results', key='clear_results')
        if clear_results:

                st.session_state.clear = True
    with col[1]:
        
        
        st.header("📈Linear Regression")
        st.subheader("Predict Exam Score")
        attendance = st.number_input("Enter Attendance (0-100):", min_value=0, max_value=100, value=50)
        input_data = pd.DataFrame({'Attendance': [attendance]})
        attendance_bins = [0, 20, 40, 60, 80, 100]
        attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        input_data['Attendance_Range'] = pd.cut(input_data['Attendance'], bins=attendance_bins, labels=attendance_labels)


        attendance_encoded = pd.get_dummies(input_data['Attendance_Range'], prefix='Attendance')
        input_data = pd.concat([input_data, attendance_encoded], axis=1)


        input_features = input_data[['Attendance_0-20', 'Attendance_21-40', 'Attendance_41-60', 'Attendance_61-80', 'Attendance_81-100']]


        prediction = model.predict(input_features)[0]
        st.write(f"**Predicted Exam Score:** {prediction:.2f}")
    
    with col[2]:

        a = df[['Hours_Studied']] 
        b = df['Exam_Score']     
        a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(a_train, b_train)
        b_pred = model.predict(a_test)
        mse = mean_squared_error(b_test, b_pred)
        r2 = r2_score(b_test, b_pred)
        hours_studied = st.number_input("Enter Hours Studied:", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        input_data = pd.DataFrame({'Hours_Studied': [hours_studied]})
        predicted_score = model.predict(input_data)[0]
        st.write(f"Predicted Average Exam Score: {predicted_score:.2f}")
        st.write(f"Model R-squared: {r2:.2f}")

    if show_dataset:
        st.subheader("Dataset")
        df

    if show_EDA:
        st.subheader("EDA")
        average_scores = df.groupby('Hours_Studied')['Exam_Score'].mean()
        st.markdown('Average Exam Score by Hours Studied')
        plt.figure(figsize=(15, 6))
        plt.bar(average_scores.index, average_scores.values)
        plt.title("Average Exam Score by Hours Studied")
        plt.xlabel("Hours Studied")
        plt.ylabel("Average Exam Score")
        plt.xticks(average_scores.index)
        plt.grid(axis='y')
        st.pyplot(plt)
        plt.clf()

        st.markdown('Peer Influence vs Exam Score')
        plt.figure(figsize=(15, 5))
        plt.scatter(df['Peer_Influence'], df['Exam_Score'])
        plt.title("Peer Influence vs Exam Score")
        plt.xlabel("Peer Influence")
        plt.ylabel("Exam Score")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()


        attendance_bins = [0, 20, 40, 60, 80, 100]
        attendance_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
        df['Attendance_Range'] = pd.cut(df['Attendance'], bins=attendance_bins, labels=attendance_labels)
        average_score = df.groupby('Attendance_Range')['Exam_Score'].mean()
        plt.figure(figsize=(15, 5))
        plt.bar(average_score.index, average_score.values)
        plt.title("Average Exam Score by Attendance Range")
        plt.xlabel("Attendance Range")
        plt.ylabel("Average Exam Score")
        plt.grid(axis='y')
        st.pyplot(plt)
        plt.clf()

    if show_columns:
        st.write(df.columns)

# Your content for the PREDICTION page goes here

# Conclusions Page

elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
                By Training a model that predicts the Average Scores of students based on their attendance and hours studied and looking at the Exploratory Data Analysis, here are the conclusions.

                #### 1. 📊Dataset Characteristics
                - The Dataset shows a relationship on all graphs. Positive Peer Influence mean higher scores according to the EDA. Same goes for the Attendance and Score. The more
                frequent the student attends the class the higher the score student gets. And for the hours studied, the more hours students put into their study hours the higher they get.

                #### 2. 🤖Model Performance (Predict Exam Score Based on Attendance.)
                - The Model has shown that the higher the attendance rate of student, the higher the chance the student will get a high score.

                #### 3. 🤖Model Performance (Predict Exam Score Based on Hours Studied)
                - The Model has consistently shown that the higher the hours student put into studying. The higher the grades that is possible for the student.

      
                """)



    # Your content for the CONCLUSION page goes here
