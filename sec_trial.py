

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Load your data
data = pd.read_csv('final_data.csv')

# Function to plot EDA
def plot_eda():
    
    # Plot histogram for 'Average Screen Time'
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Plot histogram using seaborn on the specified axes (ax)
    st.subheader("Histogram")
    sns.histplot(data=data, x='Average Screen Time', bins=20, kde=True, ax=ax)
    
    # Set titles and labels using the axes object
    ax.set_title('Histogram of Average Screen Time')
    ax.set_xlabel('Average Screen Time')
    ax.set_ylabel('Frequency')
    
    # Display the figure in Streamlit
    st.pyplot(fig)
    st.markdown("""
                ###
                - The highest frequency of screen time falls within the 10-20 range, suggesting that most people spend around this amount of time on screens.
                - The kernel density estimate indicates two peaks or modes, one around 10 and another less pronounced peak around 30, suggesting there might be two common screen time durations among the population sampled.
                """ )
    # Display the figure in Streamlit
    # Plot histogram for 'Average Spent on App (INR)'
    # plt.figure(figsize=(8, 6))
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.histplot(data=data, x='Average Spent on App (INR)', bins=20, kde=True, ax=ax1)
    ax1.set_title('Histogram of Average Spent on App (INR)')
    ax1.set_xlabel('Average Spent on App (INR)')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)
    st.markdown("""
                ### 
                - The tallest bar is located at the beginning of the histogram, which represents the lowest expenditure range. This suggests that the most common average expenditure on the app is relatively low.
                - As the average expenditure increases, the frequency of users spending that amount decreases, which is typical of consumer behavior for in-app purchases.
                Overall, the histogram indicates that while there is a range of spending behaviors on the app, the majority of users spend a lower amount of money, with fewer users making more substantial purchases.
                """)
    # Plot count plot for 'Left Review'
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=data, x='Left Review', ax=ax2)
    ax2.set_title('Count of Left Review')
    ax2.set_xlabel('Left Review')
    ax2.set_ylabel('Count')
    st.pyplot(fig2)
    st.markdown("""
                ###
                - This graph shows the count of people who left reviwes versus the count of people who did not leave the review.
                - Almost the count is same bor both classes.
                D""")
    # Plot count plot for 'Ratings'
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=data, x='Ratings', ax=ax3)
    ax3.set_title('Count of Ratings')
    ax3.set_xlabel('Ratings')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)
    st.markdown("""
                ###
                - The most frequent rating is 5.There are 140 ratings of 5.
                - Ratings of 4 and 6 are the second most frequent. There are 120 ratings of both 4 and 6.
                - Ratings become progressively less frequent as the rating value deviates from 5. There are only 20 ratings of 0 and 10.""")

    # Plot count plot for 'New Password Request'
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=data, x='New Password Request', ax=ax4)
    ax4.set_title('Count of New Password Request')
    ax4.set_xlabel('New Password Request')
    ax4.set_ylabel('Count')
    st.pyplot(fig4)
    st.markdown("""
                ###
                - The most frequent number of password requests is 1. There are more users requesting a new password once than any other number of times.
                - The number of users requesting a new password a certain number of times decreases as the number of requests increases. There are very few users requesting a new password more than 4 times.""")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    sns.histplot(data=data, x='Last Visited Minutes', bins=20, kde=True, ax=ax5)
    ax5.set_title('Histogram of Last Visited Minutes')
    ax5.set_xlabel('Last Visited Minutes')
    ax5.set_ylabel('Frequency')
    st.pyplot(fig5)
    st.markdown("""
                ###
                - The most frequent range of last visited time is between 0 and 10,000 minutes. There are more users in this range than any other.
                - The number of users decreases as the last visited time in minutes increases. There are very few users who last visited the app more than 40,000 minutes ago.""")
    # Plot count plot for 'Status'
    fig6, ax6 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=data, x='Status', ax=ax6)
    ax6.set_title('Count of Status')
    ax6.set_xlabel('Status')
    ax6.set_ylabel('Count')
    st.markdown("""
                ##
                - Most of the people have intsalled the app.""")
    st.pyplot(fig6)
    
    fig7 = go.Figure()
    for segment in data["Segments"].unique():
        fig7.add_trace(go.Scatter(x=data[data["Segments"]==segment]['Last Visited Minutes'],
                                  y=data[data["Segments"]==segment]['Average Spent on App (INR)'],
                                  mode='markers',
                                  marker=dict(size=6, line_width=1),
                                  name=str(segment)))
    
    fig7.update_layout(width=800, height=800, autosize=True, showlegend=True,
                      yaxis_title='Average Spent on App (INR)',
                      xaxis_title='Last Visited Minutes',
                      hovermode='closest')
    
    fig7.update_traces(hovertemplate='Last Visited Minutes: %{x} <br>Average Spent on App (INR): %{y}')
    
    st.plotly_chart(fig7)
    st.markdown("""
                ###
                - The blue segment shows the segment of users the app has retained over time.
                - The red segment indicates the segment of users who just uninstalled the app or are about to uninstall it soon.
                - And the green segment indicates the segment of users that the application has lost.

                """)
    st.subheader("Relationship Between Ratings and Screen Time")
    fig8 = px.scatter(data_frame=data, 
                     x="Average Screen Time",
                     y="Ratings",
                     size="Ratings",
                     color="Status",
                     
                     trendline="ols")
    
    st.plotly_chart(fig8)
    st.markdown("""
                ####
                - Users who uninstalled the app had an average screen time of fewer than 5 minutes a day, and the average spent was less than 100. 
                - We can also see a linear relationship between the average screen time and the average spending of the users still using the app.""")
    st.subheader("Relationship Between Spending Capacity and Screen Time")
    fig9= px.scatter(data_frame=data, 
                     x="Average Screen Time",
                     y="Average Spent on App (INR)",
                     size="Average Spent on App (INR)",
                     color="Status",
                     
                     trendline="ols")
    
    st.plotly_chart(fig9)
    st.markdown("""
                ###
                - So we can see that users who uninstalled the app gave the app a maximum of five ratings.
                - Their screen time is very low compared to users who rated more. So, this describes that users who don’t like to spend more time rate the app low and uninstall it at some point.""")
    
   
    st.subheader("Heatmap")
    numeric_cols = ['Average Screen Time', 'Average Spent on App (INR)', 'Last Visited Minutes', 'Ratings']
    correlation = data[numeric_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    st.pyplot(plt)
    st.markdown("""
                ###
                - There is a positive correlation between Average Screen Time and Ratings. This means that users who spend more time using the app tend to give it higher ratings.
                - There is a negative correlation between Average Screen Time and Average Spent on App (INR). This means that users who spend more time using the app tend to spend less money in the app.
                - There is no correlation between Last Visited Minutes and Average Spent on App (INR). This means that how recently a user visited the app is not related to how much money they spend in the app.""")
    st.subheader("Violin Plot of Ratings by Installation Status")
    plt.figure(figsize=(9, 5))
    sns.violinplot(x='Status', y='Ratings', data=data, palette='muted')
    st.pyplot(plt)
    st.markdown("""
                ###
                - The median rating for users with the app installed is higher than the median rating for users with the app uninstalled. This suggests that users who tend to keep the app installed are more satisfied with the app than users who uninstall it.
                - The distribution of ratings for users with the app installed is wider than the distribution of ratings for users with the app uninstalled. This suggests that there is more variation in the ratings for users who keep the app installed than for users who uninstall it.
                - The distribution of ratings for users with the app uninstalled is narrower than the distribution of ratings for users with the app installed. This suggests that there is less variation in the ratings for users who uninstall the app than for users who keep it installed.""")


    
    # Additional plots can be added here following the same pattern

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction",page_icon=":bar_chart:",layout="centered")
st.title(":bar_chart: Customer Churn Prediction")
def show_dataset():
    st.image("Customer_Churn_Prediction_Models_in_Machine_Learning.png")
    st.header("Dataset")
    st.write("Here is a glimpse of the dataset used in this dashboard.")
    
    # Show the first few rows of the dataset
    st.dataframe(data.head())

    # Add a description of the dataset
    st.markdown("""
    ## Dataset Description
    This dataset contains various user metrics that are used for predicting customer churn. The main features include:
    - **Average Screen Time**: Average daily time the user spends on the app.
    - **Average Spent on App (INR)**: Average amount of money spent by the user on the app.
    - **Left Review**: Indicates whether the user has left a review (1) or not (0).
    - **Ratings**: The user's rating for the app.
    - **New Password Request**: Indicates if the user has requested a new password (1) or not (0).
    - **Last Visited Minutes**: Minutes since the user last visited the app.
    - **Status**: Current status of the user, which could include active, inactive, etc.
    - **Segments**: User segmentation based on behavior and engagement patterns.
    """, unsafe_allow_html=True)
# Add tabs for Prediction and EDA
tabs = st.sidebar.radio("Navigation", ["Dataset","Prediction", "EDA"])
import joblib
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
# Prediction tab
# Prediction tab
if tabs == "Prediction":
    import streamlit as st
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    model = joblib.load('trained_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # Function to predict churn status
    def predict_churn(average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status ):
        # Scale the input values
        
        input_values = scaler.transform([[average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status]])
        # Predict churn status
        prediction = model.predict(input_values)
        return prediction[0]

    # Streamlit app
    #st.title('Customer Churn Prediction')

    # Input values for independent variables
    average_screen_time = st.number_input('Average Screen Time', value=0.0)
    average_spent_on_app = st.number_input('Average Spent on App (INR)', value=0.0)
    left_review = st.selectbox('Left Review', [0, 1])
    ratings = st.slider('Ratings', min_value=0, max_value=5, value=0)
    new_password_request = st.selectbox('New Password Request', [0, 1])
    last_visited_minutes = st.number_input('Last Visited Minutes', value=0.0)
    Installation_Status = st.selectbox('Installation Status', [0, 1])

    # Predict churn status
    if st.button('Predict'):
        
        prediction = predict_churn(average_screen_time, average_spent_on_app, left_review, ratings, new_password_request, last_visited_minutes,Installation_Status)
        st.write('Churn Status:', prediction)
# EDA tab
elif tabs == "EDA":
    plot_eda()
elif tabs=="Dataset":
    show_dataset()    
