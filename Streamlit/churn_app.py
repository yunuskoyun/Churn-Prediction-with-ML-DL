import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle


html_temp = """
	<div style ="background-color:#C5FFBF; padding:1px">
	<h2 style ="color:#13C800; text-align:center; ">Streamlit Employee Churn Prediction Project</h2>
	</div>
	"""

st.markdown(html_temp, unsafe_allow_html = True)
st.text('Employee Left or Stay?')

#image
image = Image.open("churns.jpg")
st.image(image, width=600)

st.header("Welcome!")
st.markdown("Please provide information of the employee on the left sidebar and than click the _Check_ button.")
st.markdown("After click the button, you will the see the employee will stay of left.")

df = pd.read_csv('HR_final.csv')



model = pickle.load(open("final_RF_pipe_model", "rb"))

#sidebar hearder
st.sidebar.header('Employee Churn Predictor')

# Departments
departments=st.sidebar.selectbox("departments", (df.departments.unique()))

# Salary
salary=st.sidebar.selectbox("salary", (df.salary.unique()))

# Satisfaction Level
satisfaction_level=st.sidebar.number_input("Satisfaction Level Score", min_value=0.00, max_value=1.00, step=0.05)

#Last Evaluation
last_evaluation = st.sidebar.number_input("Last Evaluation Score:",min_value=0.00, max_value=1.00, step=0.05)

#average_monthly_hours
average_montly_hours=st.sidebar.number_input("Average Monthly Working Hours:",min_value=10, max_value=500, step=10)

#number_project
number_project=st.sidebar.number_input("Number of Projects Worked On:",min_value=1, max_value=10, step=1)

#time_spend_company
time_spend_company=st.sidebar.number_input("Time Spend in the Company:",min_value=1, max_value=20, step=1)

radio1 = st.sidebar.radio("Received a Promotion in the Last 5 Years?:", ('Yes', 'No'))
if radio1 == 'Yes':
    promotion_last_5years = 1
else:
    promotion_last_5years = 0
    
radio2 = st.sidebar.radio("Have a work accident?:", ('Yes', 'No'))
if radio2 == 'Yes':
    work_accident = 1
else:
    work_accident = 0
      





my_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation":last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    "departments": departments, 
    "salary": salary
}

my_dict=pd.DataFrame.from_dict([my_dict])


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
enc_lab = LabelEncoder()
my_dict['departments'] = enc_lab.fit_transform(my_dict[['departments']])

enc_ord = OrdinalEncoder(categories=[['low', 'medium', 'high']])
my_dict['salary'] = enc_ord.fit_transform(my_dict[['salary']])



if st.button("Check"):
    pred = model.predict(my_dict)
    if pred[0]==0:
            st.success("_Employee will stay_")
            st.image(Image.open("stayed.jpg"), output_format="JPG", width=400)
    else:
            st.success("_Employee will left_")
            st.image(Image.open("left.jpg"), output_format="JPG", width=400)
st.sidebar.info("Please fill all required fields..")