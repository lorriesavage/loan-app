import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import streamlit.components.v1 as components
import codecs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics as mt

from PIL import Image
import io
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report


st.title("World Happiness Dashboard😁")


app_page = st.sidebar.selectbox('Select Page', ['Overview', 'Visualization', 'Prediction', 'Conclusion'])

# df = pd.read_csv("whr2023.csv")
df = pd.read_csv("whr2023.csv")
if app_page == 'Overview':
    image_path = Image.open("happiness-joy.jpg")
    st.image(image_path, width=400)
    st.write('For our project, we chose to analyze a dataset entitled, "World Happiness Report 2023 Dataset". With this dataset, we can see how key variables such as GDP per capita, social support, healthy life expectancy, freedom, generosity, and corruption, can influence happiness scores across the world.')
    
    st.subheader("Our Goals")
    st.write("The goals of our project are to analyze the factors which effect a person's happiness, and to discover how one's happiness can affect their life expectancy. We aim to understand what makes a person happy, how we can become happier, and how we can live longer.")

    st.subheader('Questions we aim to answer: ')
    st.write("- What factors most or least affect our happiness?")
    st.write("- How are life expectancy and happiness correlated?")
    st.write("- How is happiness correlated with income? Is the phrase 'money doesn't buy happiness' accurate?")
    st.write("- How does the country or region where a person lives affect their happiness?")

    st.subheader("Let's explore the dataset!")
    
    st.write("The World Happiness Report contains data from 2023. A preview of the dataset is shown below: ")
    st.dataframe(df.head())

    st.write("Information about the dataframe: ")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("Statistics on the dataset: ")
    st.dataframe(df.describe())

    st.write("Source: https://www.kaggle.com/datasets/atom1991/world-happiness-report-2023 ")


if app_page == 'Visualization':
    st.title("Data Visualization")

    list_columns = df.columns

    values = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Healthy life expectancy"])

    #Creation of the line chart
    st.subheader('Line Chart: ')
    st.line_chart(df, x=values[0], y=values[1])


    string_columns = list(df.select_dtypes(include=['object']).columns)
    data1 = df.drop(columns=string_columns)
    data = data1.drop(columns=[ 'Standard error of ladder score', 'Ladder score in Dystopia', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption', 'Dystopia + residual' ])
    #.drop(columns=...)

    # Create heatmap data (assuming 'value_column' is the column you want to visualize)
    heatmap_data = data.corr()  # Calculate correlation matrix

    # Create the heatmap
    st.subheader('Heatmap')
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", ax=ax, cmap='RdBu_r', vmin=-1, vmax=1) 
    plt.xlabel("X-axis Label", fontsize=12)
    plt.ylabel("Y-axis Label", fontsize=12)
    plt.title("Correlation Matrix", fontsize=14)
    st.pyplot(fig)

# Pairplot
    st.subheader('Pairplot:')
    values_pairplot = st.multiselect("Select 4 variables: ", list_columns, ["Healthy life expectancy", "Happiness score", "Logged GDP per capita", "Freedom to make life choices"])
    df2 = df[[values_pairplot[0], values_pairplot[1], values_pairplot[2], values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)


    st.subheader('Link to Report:')
    st.write('https://colab.research.google.com/drive/1K1Sx4b_Hv8UGfeC7RRgI9Sdx3D-zPIw2#scrollTo=u2TMKY7GaVpF')

    #profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    #st_profile_report(profile)
    
    st.write('Map of Global Happiness for 2023, based off of World Happiness Report Dataset 2023, courtesy of Visual Capitalist.')
    image2 = Image.open('worlds-happiest-countries-2023-MAIN-1.jpg')
    st.image(image2, width=400)

    st.write('Map of global GDP for 2022, courtesy of World Bank: ')
    image3 = Image.open('gdp-per-capita-worldbank.jpg')
    st.image(image3, width=400)


list_columns = df.columns


if app_page == 'Prediction':
    st.title("Prediction")

    list_columns = df.columns

    input_lr = st.multiselect("Select two variables: ", list_columns, ["Happiness score", "Logged GDP per capita"])

    df_new = df.dropna()
    df2 = df_new[input_lr]

    X = df2
    y = df_new["Healthy life expectancy"]

    col1, col2 = st.columns(2)
    col1.subheader("Feature Columns top 25 ")
    col1.write(X.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    explained_variance = np.round(mt.explained_variance_score(y_test, pred) * 100, 2)
    mae = np.round(mt.mean_absolute_error(y_test, pred), 2)
    mse = np.round(mt.mean_squared_error(y_test, pred), 2)
    r_square = np.round(mt.r2_score(y_test, pred), 2)

    # Create a comparison DataFrame to visualize Actual vs Predicted values
    comparison_df = pd.DataFrame({'Actual Values': y_test, 'Predicted Values': pred})

    # Display the first 10 rows of the comparison DataFrame
    st.write("### Comparison of Actual vs. Predicted Values")
    st.write(comparison_df.head(10))
    
    # Display results
    st.subheader('🎯 Results')
    st.write("1) The model explains,", explained_variance, "% variance in healthy life expectancy.")
    st.write("2) The Mean Absolute Error of the model is:", mae, "indicating the average error in predictions.")
    st.write("3) MSE: ", mse, "reflecting the average squared differences from the actual values.")
    st.write("4) The R-Square score of the model is", r_square, "suggesting that the model captures a moderate portion of the variability in life expectancy.")



    # Feature Importance Analysis
    st.subheader("📊 Feature Importance")
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': input_lr,
        'Coefficient': lr.coef_
    })
    
    # Calculate absolute importance
    importance_df['Absolute Importance'] = np.abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values(by='Absolute Importance', ascending=False)

    # Display the feature importance DataFrame
    st.write(importance_df)

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance in Predicting Healthy Life Expectancy')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Features')
    st.pyplot(plt)  # Display the plot in Streamlit

    # Plotting the Linear Regression line
    st.subheader('📈 Linear Regression Line')

    # Create a scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df_new[input_lr[0]], y=y, data=df_new, scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})
    plt.title(f'Linear Regression of Healthy Life Expectancy vs {input_lr[0]}')
    plt.xlabel(input_lr[0])
    plt.ylabel('Healthy Life Expectancy')
    
    st.pyplot(plt)  # Display the plot in Streamlit




if app_page == 'Conclusion':

    st.title('Conclusion')
    st.balloons()

    st.subheader('1. Insights:')
    st.markdown('- Correlation Analysis: Our heatmap and pairplot analyses revealed significant correlations between happiness and other variables. Notably, factors such as healthy life expectancy and GDP per capita had strong positive correlations with happiness scores, which confirms the importance of both wealth and healthiness in increased happiness.')
    st.markdown('- Life Expectancy and Happiness: The linear regression model also demonstrated that healthy life expectancy could be predicted with a reasonable degree of accuracy using features like GDP per capita and happiness score. This shows a strong link between how long people live and their overall well-being.')
    st.subheader('2. Model Performance: ')
    st.markdown("- Our predictive model for healthy life expectancy, using a simple linear regression model, achieved an explained variance score of approximately 72%. This suggests that while our model captures a substantial portion of the variability in life expectancy, there is room for improvement in the model's performance.")
    st.markdown("- The Mean Absolute Error (MAE) and Mean Squared Error (MSE) values were acceptable, indicating that the model predictions were close to the actual values. However, future improvements could focus on enhancing the model by incorporating additional features or using more complex models.")
    st.subheader('3. Ways to Improve Model: ')
    st.markdown("- Data Quality and Feature Engineering: Though our current dataset provides a comprehensive look at happiness factors, there are still missing values in some areas. Handling these more effectively, potentially by using imputation strategies or by introducing new variables like mental health or social safety nets, could enhance the model's predictive abilities.")
    st.subheader('4. Longterm Considerations: ')
    st.markdown("- Dynamic Updates: Happiness and well-being are dynamic, influenced by changes in political, economic, and environmental conditions. Continuously updating the model with more recent data, such as future World Happiness Reports, would ensure the model remains relevant and accurate.")
    st.markdown("- Integrating Additional Dataset: To further enhance the depth of analysis, integrating extra datasets, such as those on mental health, education, or environmental factors, could provide new perspectives on what drives happiness around the world.")
    


