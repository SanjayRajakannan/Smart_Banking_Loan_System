import streamlit as st
import plotly.express as px
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
st.set_page_config(
    page_title="Smart Banking System",
    page_icon="🏦",
    layout="wide"
)
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

h1{
    color:#4CAF50;
}

[data-testid="stMetricValue"]{
    color:#00FF88;
    font-size:30px;
}

[data-testid="stMetricLabel"]{
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)
st.sidebar.title("🏦 Smart Banking")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Loan Prediction",
        "Analytics",
        "Customer Segments",
        "EMI Calculator",
        "Model Performance",
        "About"
    ]
)


if page == "Dashboard":
    st.image("images/bank_pic.png", width=130)
    st.title(" Smart Banking Dashboard")

st.markdown("### Welcome to the Smart Banking Loan Approval System")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 Total Customers", "614")

with col2:
    st.metric("✅ Approved Loans", "422")

with col3:
    st.metric("❌ Rejected Loans", "192")

with col4:
    st.metric("🎯 Accuracy", "78.86%")

st.divider()

st.subheader("Project Overview")

st.write("""
This Smart Banking System predicts whether a loan should be approved or rejected using Machine Learning.

### Algorithms Used
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors 
- K-Means Clustering
- Principal Component Analysis 

Use the navigation menu on the left to explore the system.
""")

if page == "Loan Prediction":

    st.title("🏦 Smart Banking Loan Approval Prediction")
    st.markdown("### Enter Customer Details")

    st.divider()

    
    left, right = st.columns(2)

   

    with left:

        st.subheader("👤 Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Marital Status",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Number of Dependents",
            [0, 1, 2, 3]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"]
        )

    
    with right:

        st.subheader("💰 Financial Information")

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=150
        )

        loan_term = st.number_input(
            "Loan Term (Months)",
            min_value=0,
            value=360
        )

        credit_history = st.selectbox(
            "Credit History",
            ["Good", "Bad"]
        )

    st.divider()

    

    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0

    credit_history = 1 if credit_history == "Good" else 0

    if property_area == "Rural":
        property_area = 0
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 2


    if st.button("🔍 Predict Loan Status", use_container_width=True):

        customer = np.array([[
            0,                      # Loan_ID
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]])

        prediction = model.predict(customer)

        st.divider()

        if prediction[0] == 1:

            st.success("✅ Congratulations! Loan Approved")

            st.info("🟢 Risk Level : LOW")

            st.metric(
                "Approval Probability",
                "High"
            )

            st.balloons()

        else:

            st.error("❌ Sorry! Loan Rejected")

            st.warning("🔴 Risk Level : HIGH")

            st.metric(
                "Approval Probability",
                "Low"
            )

        st.divider()

        st.subheader("📋 Customer Summary")

        st.write(f"**Gender:** {('Male' if gender == 1 else 'Female')}")
        st.write(f"**Married:** {('Yes' if married == 1 else 'No')}")
        st.write(f"**Dependents:** {dependents}")
        st.write(f"**Education:** {('Graduate' if education == 0 else 'Not Graduate')}")
        st.write(f"**Applicant Income:** ₹{applicant_income:,}")
        st.write(f"**Coapplicant Income:** ₹{coapplicant_income:,}")
        st.write(f"**Loan Amount:** ₹{loan_amount:,}")
        st.write(f"**Loan Term:** {loan_term} Months")

elif page == "Analytics":

    st.title("📊 Analytics")

    import pandas as pd
    import plotly.express as px

    
    data = pd.read_csv("loan_data.csv")

    
    loan_counts = data["Loan_Status"].value_counts()

    fig = px.pie(
        values=loan_counts.values,
        names=loan_counts.index,
        title="Loan Approval Distribution"
    )

    st.plotly_chart(fig)

   
    property_counts = data["Property_Area"].value_counts()

    fig = px.bar(
        x=property_counts.index,
        y=property_counts.values,
        title="Property Area Distribution"
    )

    st.plotly_chart(fig)

    fig = px.histogram(
        data,
        x="ApplicantIncome",
        title="Applicant Income Distribution"
    )

    st.plotly_chart(fig)



elif page == "Customer Segments":

    st.title("👥 Customer Segmentation")
    st.markdown("### Customer Segmentation using K-Means Clustering")

    import pandas as pd
    import plotly.express as px
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    
    data = pd.read_csv("loan_data.csv")

    
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)

    
    encoder = LabelEncoder()

    object_columns = data.select_dtypes(include=['object', 'string']).columns

    for column in object_columns:
        data[column] = encoder.fit_transform(data[column])

    
    X = data.drop("Loan_Status", axis=1)

     
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X_scaled)

    data["Cluster"] = clusters

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PCA1": pca_data[:,0],
        "PCA2": pca_data[:,1],
        "Cluster": data["Cluster"].astype(str)
    })

    
    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="Customer Segmentation using K-Means",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    
    st.subheader("📊 Customers in Each Segment")

    cluster_counts = data["Cluster"].value_counts().sort_index()

    cluster_table = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Customers": cluster_counts.values
    })

    st.dataframe(cluster_table, use_container_width=True)

    st.divider()

    fig2 = px.pie(
        cluster_table,
        values="Customers",
        names="Cluster",
        title="Customer Segment Distribution",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("📌 Cluster Explanation")

    st.success("""
🟢 Cluster 0

• Low Risk Customers

• Good Credit History

• High Chance of Loan Approval
""")

    st.warning("""
🟡 Cluster 1

• Medium Risk Customers

• Average Income

• Moderate Approval Probability
""")

    st.error("""
🔴 Cluster 2

• High Risk Customers

• Poor Credit History

• Low Chance of Loan Approval
""")

elif page == "EMI Calculator":

    st.title("💰 EMI Calculator")

    principal = st.number_input(
        "Loan Amount",
        value=100000
    )

    rate = st.number_input(
        "Interest Rate (%)",
        value=8.5
    )

    years = st.number_input(
        "Loan Duration (Years)",
        value=5
    )

    if st.button("Calculate EMI"):

        r = rate / (12 * 100)

        n = years * 12

        emi = (
            principal * r * (1+r)**n
        ) / (
            (1+r)**n - 1
        )

        st.success(
            f"Monthly EMI = ₹{emi:.2f}"
        )

elif page == "Model Performance":

    st.title("📈 Model Performance")

    import pandas as pd
    import plotly.express as px

    model_data = {
        "Algorithm": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ],

        "Accuracy": [
            78.86,
            71.54,
            78.05,
            78.05
        ]
    }

    df = pd.DataFrame(model_data)

    st.subheader("Accuracy Table")

    st.dataframe(df)

    fig = px.bar(
        df,
        x="Algorithm",
        y="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig)
elif page == "About":

    st.title("ℹ️ About")

    st.write("""
### 🏦 Smart Banking & Loan Approval System

This project was developed to make the loan approval process simpler and more efficient using Machine Learning. It predicts whether a loan application is likely to be approved based on customer details such as income, credit history, education, and loan amount.

### >> Features
- Loan Approval Prediction
- Customer Segmentation
- EMI Calculator
- Analytics Dashboard
- Model Performance Comparison

### >> Machine Learning Algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Principal Component Analysis (PCA)

### >> Technologies Used
Python, Streamlit, Pandas, NumPy, Scikit-learn, and Plotly.

### 👨‍💻 Developed By
*** DATA DYNAMOS ***
B.Tech Information Technology
""")
    
    fig = px.pie(
    values=[422,192],
    names=["Approved","Rejected"],
    title="Loan Status Distribution"
)

