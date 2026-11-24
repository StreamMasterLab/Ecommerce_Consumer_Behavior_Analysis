# ============================================================
# E-COMMERCE CONSUMER BEHAVIOR ANALYSIS
# STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Consumer Analysis",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_engineered_data.csv"
)

SEGMENTATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "customer_segmentation.pkl"
)

PURCHASE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "purchase_prediction.pkl"
)

SPENDING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spending_prediction.pkl"
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():

    segmentation_model = joblib.load(
        SEGMENTATION_MODEL_PATH
    )

    purchase_model = joblib.load(
        PURCHASE_MODEL_PATH
    )

    spending_model = joblib.load(
        SPENDING_MODEL_PATH
    )

    return (
        segmentation_model,
        purchase_model,
        spending_model
    )


df = load_data()

(
    segmentation_model,
    purchase_model,
    spending_model
) = load_models()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 E-Commerce Analysis")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Spending Prediction",
        "Purchase Behavior",
        "Customer Segmentation"
    ]
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "🛒 E-Commerce Consumer Analysis"
    )

    st.write(
        "Analyze customer spending, purchase behavior "
        "and customer segments using Machine Learning."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_customers = df.shape[0]

    total_spending = df["Purchase_Amount"].sum()

    average_spending = df["Purchase_Amount"].mean()

    average_satisfaction = (
        df["Customer_Satisfaction"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        total_customers
    )

    col2.metric(
        "Total Spending",
        f"${total_spending:,.2f}"
    )

    col3.metric(
        "Average Spending",
        f"${average_spending:,.2f}"
    )

    col4.metric(
        "Avg Satisfaction",
        f"{average_satisfaction:.2f}"
    )

    st.divider()

    # --------------------------------------------------------
    # PURCHASE CATEGORY
    # --------------------------------------------------------

    st.subheader(
        "Purchase Category Analysis"
    )

    category_data = (
        df.groupby("Purchase_Category")
        ["Purchase_Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_data)

    # --------------------------------------------------------
    # PURCHASE CHANNEL
    # --------------------------------------------------------

    st.subheader(
        "Purchase Channel Analysis"
    )

    channel_data = (
        df.groupby("Purchase_Channel")
        ["Purchase_Amount"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(channel_data)

    # --------------------------------------------------------
    # PURCHASE INTENT
    # --------------------------------------------------------

    st.subheader(
        "Purchase Intent Distribution"
    )

    intent_data = (
        df["Purchase_Intent"]
        .value_counts()
    )

    st.bar_chart(intent_data)

# ============================================================
# SPENDING PREDICTION
# ============================================================

elif page == "Spending Prediction":

    st.title(
        "💰 Customer Spending Prediction"
    )

    st.write(
        "Predict the expected purchase amount "
        "for a customer."
    )

    st.divider()

    st.info(
        "Enter customer information below."
    )

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        frequency = st.number_input(
            "Frequency of Purchase",
            min_value=1,
            value=5
        )

        brand_loyalty = st.number_input(
            "Brand Loyalty",
            min_value=0.0,
            value=5.0
        )

        product_rating = st.number_input(
            "Product Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0
        )

        research_time = st.number_input(
            "Product Research Time (hours)",
            min_value=0.0,
            value=2.0
        )

    with col2:

        return_rate = st.number_input(
            "Return Rate",
            min_value=0.0,
            value=0.1
        )

        satisfaction = st.number_input(
            "Customer Satisfaction",
            min_value=0.0,
            max_value=10.0,
            value=7.0
        )

        decision_time = st.number_input(
            "Time to Decision",
            min_value=0.0,
            value=2.0
        )

        purchase_month = st.number_input(
            "Purchase Month",
            min_value=1,
            max_value=12,
            value=6
        )

        purchase_day = st.number_input(
            "Purchase Day of Week",
            min_value=0,
            max_value=6,
            value=2
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predict Spending",
        type="primary"
    ):

        engagement_score = (
            brand_loyalty +
            satisfaction +
            product_rating
        ) / 3

        spending_frequency = (
            1
        )

        research_decision_ratio = (
            research_time /
            (decision_time + 1)
        )

        input_data = pd.DataFrame({

            "Age": [age],

            "Frequency_of_Purchase": [
                frequency
            ],

            "Brand_Loyalty": [
                brand_loyalty
            ],

            "Product_Rating": [
                product_rating
            ],

            "Time_Spent_on_Product_Research(hours)": [
                research_time
            ],

            "Return_Rate": [
                return_rate
            ],

            "Customer_Satisfaction": [
                satisfaction
            ],

            "Time_to_Decision": [
                decision_time
            ],

            "Purchase_Month": [
                purchase_month
            ],

            "Purchase_DayOfWeek": [
                purchase_day
            ],

            "Purchase_Hour": [
                12
            ],

            "Spending_per_Purchase_Frequency": [
                spending_frequency
            ],

            "Engagement_Score": [
                engagement_score
            ],

            "Research_to_Decision_Ratio": [
                research_decision_ratio
            ]

        })

        # Add missing categorical values
        for column in spending_model.feature_names_in_:

            if column not in input_data.columns:

                input_data[column] = (
                    df[column].mode()[0]
                    if column in df.columns
                    else 0
                )

        input_data = input_data[
            spending_model.feature_names_in_
        ]

        prediction = spending_model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Purchase Amount: "
            f"${prediction:,.2f}"
        )

# ============================================================
# PURCHASE BEHAVIOR
# ============================================================

elif page == "Purchase Behavior":

    st.title(
        "🛒 Purchase Behavior Prediction"
    )

    st.write(
        "Predict the customer's purchase intent."
    )

    st.divider()

    st.info(
        "The trained classification model predicts "
        "the Purchase Intent category."
    )

    # --------------------------------------------------------
    # SELECT CUSTOMER
    # --------------------------------------------------------

    customer_index = st.number_input(
        "Select Customer Row",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    customer = df.iloc[
        int(customer_index)
    ]

    st.subheader(
        "Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Age",
        int(customer["Age"])
    )

    col2.metric(
        "Purchase Frequency",
        int(customer["Frequency_of_Purchase"])
    )

    col3.metric(
        "Satisfaction",
        round(
            customer["Customer_Satisfaction"],
            2
        )
    )

    if st.button(
        "Predict Purchase Behavior",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [customer]
        )

        input_data = input_data.drop(
            columns=[
                "Purchase_Intent",
                "Purchase_Amount",
                "Time_of_Purchase"
            ],
            errors="ignore"
        )

        # Ensure columns match trained model
        input_data = input_data[
            [
                column
                for column in purchase_model.feature_names_in_
                if column in input_data.columns
            ]
        ]

        prediction = purchase_model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Purchase Intent: {prediction}"
        )

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":

    st.title(
        "👥 Customer Segmentation"
    )

    st.write(
        "Explore different customer groups "
        "created using K-Means clustering."
    )

    st.divider()

    if "Customer_Segment" in df.columns:

        segment_counts = (
            df["Customer_Segment"]
            .value_counts()
            .sort_index()
        )

        st.subheader(
            "Customers in Each Segment"
        )

        st.bar_chart(
            segment_counts
        )

        # ----------------------------------------------------
        # SEGMENT SUMMARY
        # ----------------------------------------------------

        segmentation_features = (
            segmentation_model["features"]
        )

        segment_summary = (
            df.groupby(
                "Customer_Segment"
            )[segmentation_features]
            .mean()
            .round(2)
        )

        st.subheader(
            "Segment Characteristics"
        )

        st.dataframe(
            segment_summary,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SELECT SEGMENT
        # ----------------------------------------------------

        selected_segment = st.selectbox(
            "Select Customer Segment",
            sorted(
                df["Customer_Segment"]
                .unique()
            )
        )

        segment_customers = df[
            df["Customer_Segment"]
            == selected_segment
        ]

        st.write(
            f"Customers in Segment "
            f"{selected_segment}: "
            f"{len(segment_customers)}"
        )

        st.subheader(
            "Selected Segment Spending"
        )

        st.metric(
            "Average Purchase Amount",
            f"${segment_customers['Purchase_Amount'].mean():,.2f}"
        )

        st.metric(
            "Average Purchase Frequency",
            f"{segment_customers['Frequency_of_Purchase'].mean():.2f}"
        )

    else:

        st.warning(
            "Customer segmentation data not found."
        )
