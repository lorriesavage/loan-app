import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

#import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from shapash.explainer.smart_explainer import SmartExplainer
import io
from PIL import Image

import mlflow
from mlflow import log_metric, log_param, log_artifact
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib


st.title("💳💸📈Credit Risk📉💸💳")

try:
    df = pd.read_csv("credit.csv")
except FileNotFoundError:
    st.error("The dataset file 'credit.csv' was not found. Please ensure it's in the correct path.")

app_page = st.sidebar.selectbox('Select Page', ['Business Case Presentation and Data Description', 'Data Visualization', 'Deployment','Prediction Models', 'Feature Importance and Driving Variables', 'Hyperparameter Tuning Experiences and Best Performing Model', 'Conclusion'])



st.sidebar.header("Automated Data Preprocessing")
drop_unknowns = st.sidebar.checkbox("Drop rows with unknown values", value=True)
    
# Columns to encode
columns_to_encode = [
    "checking_balance", "credit_history", "savings_balance", 
    "employment_duration", "housing", "job", "phone", "default", "other_credit"
]

# Drop columns
if "purpose" in df.columns:
    df.drop("purpose", axis=1, inplace=True)
if "phone" in df.columns:
    df.drop("phone", axis=1, inplace = True)
if "housing" in df.columns:
    df.drop("housing", axis=1, inplace=True )
if "employment_duration" in df.columns:
    df.drop("employment_duration", axis=1, inplace=True )

# Encode the specified columns
label_encoders = {}
for col in columns_to_encode:
    if col in df.columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

# Handle unknown values
if drop_unknowns:
    df.replace("unknown", np.nan, inplace=True)
    df.dropna(inplace=True)


if app_page == 'Business Case Presentation and Data Description':
    image_path = Image.open("money-lender-1170x710.webp")
    st.image(image_path, width=400)
    st.header('Overview')

    st.subheader('Project Summary:')
    st.write('This app utilizes machine learning algorithms to predict the likelihood of a loan applicant defaulting on their loan.')


    st.subheader('Model Performance:')
    st.write('We tested two models:')
    st.write('- **KNN (K-Nearest Neighbors)**: Achieved an accuracy of **0.65**.')
    st.write('- **Decision Tree**: Achieved an accuracy of **0.68**.')

    st.write('Although the models show moderate performance, they still provide valuable insights into the factors that influence loan default and could help in making data-driven lending decisions.')

    st.subheader('Business Problem:')
    st.write('The goal of this project was to **optimize the loan repayment schedule** based on the financial data of applicants. By predicting the likelihood of default, financial institutions can offer tailored repayment plans that ensure timely payments, reducing the risk of defaults. This will help both institutions and customers by creating more sustainable financial agreements.')

    st.write('Here is the dataset we will work with: ')
    st.dataframe(df)
    st.write("Source: https://www.kaggle.com/datasets/daniellopez01/credit-risk/data ")

    st.write('Header of dataset: ')
    st.dataframe(df.head())

    st.write("Information about the dataframe: ")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Statistics on the dataset: ")
    st.dataframe(df.describe())



if app_page == 'Data Visualization':
    st.header('Graphs and stuff')
    data = {'credit_history': ['critical', 'poor', 'good', 'very good', 'perfect']}
    df2 = pd.DataFrame(data)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['ch-encoded'] = label_encoder.fit_transform(df['credit_history']) 
    #st.dataframe(df.head())

    list_columns = df.columns
    values = st.multiselect("Select two variables: ", list_columns, ["amount", "percent_of_income"])

    
    if len(values) == 2:
        st.line_chart(df[values])


    # Dropdowns for selecting variables
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    selected_x = st.selectbox("Select a categorical variable for the x-axis:", numeric_columns)
    selected_y = st.selectbox("Select a numeric variable for the y-axis:", numeric_columns)

    # Check if user has selected both variables
    if selected_x and selected_y:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.swarmplot(data=df, x=selected_x, y=selected_y, palette='pastel', ax=ax)
            st.pyplot(fig)
        except ValueError as e:
            st.error(f"Error creating swarmplot: {e}")

    show_pie_chart = st.checkbox("Show Pie Chart")

    if show_pie_chart:
        var = st.selectbox("Select a column for Pie Chart:", numeric_columns)
        st.subheader(f"Pie Chart of {var} Distribution")
        pie_data = df[var].value_counts()  # Adjust to your dataset column
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(pie_data)))
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

X = df.drop(["default"], axis=1)
y = df.default
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)

tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
prediction2 = tree.predict(X_test)
y_pred = pd.Series(prediction)

if app_page == 'Prediction Models':
    MODELS = {
        #"Logistic Regression":LogisticRegression,
        "KNN":KNeighborsClassifier,
        "Decision Tree":DecisionTreeClassifier,
    }

    model_mode=st.sidebar.selectbox("Select a model of your choice",['KNN','Decision Tree'])


    #X = df.drop(["default"], axis=1)
    #y = df.default

    #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


    if model_mode == 'KNN':
        knn=KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train,y_train)
        prediction = knn.predict(X_test)
        st.write("Accuracy KNN:",metrics.accuracy_score(y_test,prediction))


    elif model_mode == 'Decision Tree':
        tree=DecisionTreeClassifier()
        tree.fit(X_train,y_train)
        prediction = tree.predict(X_test)
        st.write("Accuracy Tree:",metrics.accuracy_score(y_test,prediction))



if app_page == 'Feature Importance and Driving Variables':
    from shapash.explainer.smart_explainer import SmartExplainer
    X_test_reset = X_test.reset_index(drop=True)
    explainer = SmartExplainer(model=tree)
    explainer.compile(x=X_test_reset, y_pred=y_pred)
    importance_plot = explainer.plot.features_importance() 
    st.plotly_chart(importance_plot, use_container_width=True)

if app_page == 'Hyperparameter Tuning Experiences and Best Performing Model':
    st.header('Parameter Tuning and ML Flow')

    import dagshub
    dagshub.init(repo_owner='lorriesavage', repo_name='finalProj', mlflow=True)

    # Start MLflow run
    with mlflow.start_run():

        mlflow.log_param("parameter name","value")
        mlflow.log_metric("Accuracy",0.9)

        mlflow.end_run()


        st.title("ML Flow Visualization")

        ui.link_button(text="👉 Go to ML Flow",url="__________[put your dagshub link here pls]",key="link_btnmlflow")


        X = df.drop(["default"], axis=1)
        y = df.default

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create a decision tree classifier
        dt = DecisionTreeClassifier(random_state=42)

        # Define a parameter grid to search over
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

        # Perform grid search to find the best parameters
        grid_search.fit(X_train, y_train)

        # Log the best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Evaluate the model
        best_dt = grid_search.best_estimator_
        test_score = best_dt.score(X_test, y_test)

        # Log the performance metric
        y_pred = pd.Series(prediction)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)

        log_metric("accuracy", accuracy)
        log_metric("precision", precision)
        log_metric("recall", recall)
        log_metric("f1", f1)


        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        # Log the best model in MLflow
        mlflow.sklearn.log_model(best_dt, "best_dt")

        # Save the model to the MLflow artifact store
        mlflow.sklearn.save_model(best_dt, "best_dt_model")



    
if app_page == 'Deployment':
    st.title("🎯Model Deployment🎯")

    model_choice = st.selectbox('Choose a model:', ['KNN', 'Decision Tree'])  # Choose model

    # User inputs for prediction
    feature_names = ['age', 'savings_balance', 'credit_history', 'amount', 'percent_of_income', 'existing_loan_count', 'checkings_balance', 'months_loan_duration', 'years_at_residence', 'other_credit', 'job', 'dependents']
    input_data = []

    for feature_name in feature_names:
        feature_value = st.number_input(f'Enter {feature_name}:', min_value=0, step=1, format="%d")  # Only whole numbers allowed
        input_data.append(feature_value)

    # When the user clicks 'Predict', use the selected model to make a prediction
    if st.button('Predict'):
        input_data_array = np.array(input_data).reshape(1, -1)

        if model_choice == 'KNN':
            prediction = knn.predict(input_data_array)
        else:  # Decision Tree
            prediction = tree.predict(input_data_array)

        # Output the result
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'
        if prediction_label == 'Yes':
            st.success(f'Prediction: {prediction_label}! You will get the loan')
        else:
            st.success(f'Prediction: You will not get the loan :(')



if app_page == 'Conclusion':
    st.balloons()

    image1 = Image.open("credit-risk-overview.png")
    st.image(image1, width=400)

    st.markdown("""
    ## In Conclusion...
    This project was all about predicting loan defaults to help financial institutions make smarter, data-driven decisions. 
    We analyzed key features such as:
    - **Checking Balance**: Provides insight into an applicant's financial stability.
    - **Credit History**: Helps evaluate creditworthiness based on past borrowing behavior.
    - **Savings Balance**: Indicates financial security and repayment potential.
    - **Employment Duration**: Longer employment suggests greater repayment ability.
    - **Existing Loans Count**: Highlights the risk associated with multiple loans.

    Using these features, we built models to predict whether a loan applicant might default. Among the models tested:
    - **KNN** achieved an accuracy of **64.67%**.
    - **Decision Tree** achieved an accuracy of **68%**.

    ## Why This Matters
    These results highlight how machine learning can support smarter loan decision-making. By analyzing applicants' financial data, these models can help institutions:
    - Assess risks more effectively.
    - Improve repayment schedules.
    - Avoid defaults.
    This leads to better outcomes for both lenders and borrowers.

    ## Looking Ahead
    While our models performed well, there's always room to grow. For example:
    - **Add More Features**: Incorporating additional data like repayment history or credit utilization could make predictions even more precise.
    - **Focus on Real-World Applications**: Aligning model predictions with actionable insights, such as flagging at-risk borrowers early, can make the system more practical.

    This project is just the beginning of how machine learning can revolutionize lending decisions. With more data, advanced techniques, and continuous improvement, we can build even more reliable systems to make loan approvals fairer, faster, and smarter.

    Thank you for your time and attention!👋
    """)
