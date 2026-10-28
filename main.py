import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="REAL LIFE RETAIL SALES",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# 2. CUSTOM LIGHT THEME AND FRONTEND
# --------------------------------------------------

st.markdown("""
<style>

/* Main Application Background */
.stApp {
    background: linear-gradient(
        135deg,
        #F5F3FF 0%,
        #F8FAFF 50%,
        #EEF6FF 100%
    );
    color: #243047;
}

/* Main Content Spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Top Header */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #5B4BCE;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.subtitle {
    color: #64748B;
    font-size: 17px;
    margin-bottom: 28px;
}

/* Section Titles */
.section-title {
    color: #312E81;
    font-size: 25px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 18px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EDE9FE;
    border-right: 1px solid #DDD6FE;
}

[data-testid="stSidebar"] * {
    color: #312E81;
}

/* Form Card */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 24px;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(79, 70, 229, 0.07);
}

/* Input Labels */
.stApp label {
    color: #475569 !important;
    font-weight: 600;
}

/* Text Input */
div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1;
    border-radius: 9px;
}

/* Input Text Color */
div[data-baseweb="input"] input {
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
}

div[data-baseweb="select"] * {
    color: #1E293B;
}

/* Focused Inputs */
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.12);
}

/* Number Input Buttons */
div[data-testid="stNumberInput"] button {
    background-color: #F1F5F9;
    color: #4F46E5;
    border: none;
}

/* Slider */
div[data-testid="stSlider"] {
    color: #7C3AED;
}

/* Predict Button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(
        90deg,
        #7C3AED,
        #4F46E5
    );
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    padding: 13px 22px;
    font-weight: 700;
    min-height: 48px;
    transition: all 0.2s ease;
}

/* Button Text */
div[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF !important;
}

/* Button Hover */
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(
        90deg,
        #6D28D9,
        #4338CA
    );
    box-shadow: 0 5px 14px rgba(79, 70, 229, 0.22);
    transform: translateY(-1px);
}

/* Other Buttons */
div.stButton > button {
    background-color: #7C3AED;
    color: #FFFFFF;
    border-radius: 9px;
    border: none;
}

/* General Subheaders */
.stApp h2,
.stApp h3 {
    color: #312E81;
    font-weight: 750;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E0E7FF;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.06);
}

/* Metric Labels */
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

/* Metric Values */
[data-testid="stMetricValue"] {
    color: #5B4BCE !important;
    font-weight: 750;
}

/* Success Alert */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Dataframe Container */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* Horizontal Divider */
hr {
    border-color: #DDD6FE;
    margin-top: 28px;
    margin-bottom: 28px;
}

/* General Markdown Text */
.stMarkdown p {
    line-height: 1.7;
}

/* Responsive Heading */
@media (max-width: 768px) {
    .main-title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 15px;
    }

    .section-title {
        font-size: 21px;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(
        BASE_DIR / "xgboost_retail_model.joblib"
    )

    preprocessor = joblib.load(
        BASE_DIR / "retail_preprocessor.joblib"
    )

except Exception as e:
    st.error(
        f"Error loading model or preprocessor: {e}"
    )
    st.stop()


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🛍️ REAL LIFE RETAIL SALES'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer purchase channel prediction '
    'to support smarter marketing decisions.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 5. CUSTOMER INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🛒 Customer & Basket Profile'
    '</div>',
    unsafe_allow_html=True
)

with st.form("retail_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        Customer_Segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"]
        )

        Region = st.selectbox(
            "Region",
            ["North", "South", "East", "West"]
        )

        City = st.selectbox(
            "City",
            [
                "Mumbai",
                "Delhi",
                "Bangalore",
                "Pune",
                "Chennai",
                "Kolkata"
            ]
        )

    with col2:

        Category = st.selectbox(
            "Category",
            [
                "Electronics",
                "Clothing",
                "Home & Kitchen",
                "Beauty",
                "Sports"
            ]
        )

        Product = st.text_input(
            "Product Name",
            value="Laptop"
        )

        Quantity = st.number_input(
            "Quantity",
            min_value=1,
            value=1,
            step=1
        )

    with col3:

        Unit_Price_INR = st.number_input(
            "Unit Price (INR)",
            min_value=0.0,
            value=25000.0,
            step=500.0
        )

        Discount_Percent = st.slider(
            "Proposed Discount (%)",
            min_value=0,
            max_value=100,
            value=10
        ) / 100.0

    submitted = st.form_submit_button(
        "🔍 Predict Purchase Channel",
        use_container_width=True
    )


# --------------------------------------------------
# 6. PREDICTION LOGIC
# --------------------------------------------------

if submitted:

    # Calculate transaction values
    sales = (
        Unit_Price_INR
        * Quantity
        * (1 - Discount_Percent)
    )

    cost = Unit_Price_INR * Quantity * 0.70

    profit = sales - cost

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Customer_Segment": [Customer_Segment],
        "Region": [Region],
        "City": [City],
        "Category": [Category],
        "Product": [Product],
        "Quantity": [Quantity],
        "Unit_Price_INR": [Unit_Price_INR],
        "Discount_Percent": [Discount_Percent],
        "Sales_INR": [sales],
        "Cost_INR": [cost],
        "Profit_INR": [profit]
    })

    try:

        # Transform input data
        transformed_data = preprocessor.transform(
            input_data
        )

        # Predict purchase channel
        prediction = model.predict(
            transformed_data
        )[0]

        probabilities = model.predict_proba(
            transformed_data
        )[0]

        # Identify the class labels
        classes = list(model.classes_)

        digital_index = classes.index(1)
        store_index = classes.index(0)

        digital_probability = (
            probabilities[digital_index] * 100
        )

        store_probability = (
            probabilities[store_index] * 100
        )

        # ------------------------------------------
        # 7. DISPLAY PREDICTION
        # ------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📊 Propensity Verdict'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.success(
                "📱 DIGITAL BUYER "
                "(Online / Mobile App)"
            )

            st.write(
                "The model predicts that this customer "
                "is more likely to purchase digitally."
            )

            st.write(
                "**Suggested Action:** Consider online "
                "advertising, email campaigns, and "
                "digital retargeting."
            )

        else:

            st.warning(
                "🏬 PHYSICAL RETAIL BUYER (In-Store)"
            )

            st.write(
                "The model predicts that this customer "
                "is more likely to purchase in a physical store."
            )

            st.write(
                "**Suggested Action:** Consider in-store "
                "promotions, SMS campaigns, and store coupons."
            )

        # ------------------------------------------
        # 8. DISPLAY PROBABILITIES
        # ------------------------------------------

        st.subheader(
            "Purchase Channel Probabilities"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🏬 In-Store Probability",
                value=f"{store_probability:.2f}%"
            )

        with col2:
            st.metric(
                label="📱 Digital Probability",
                value=f"{digital_probability:.2f}%"
            )

        # ------------------------------------------
        # 9. DISPLAY INPUT SUMMARY
        # ------------------------------------------

        st.subheader("Customer Input Summary")

        st.dataframe(
            input_data,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )

        st.info(
            "Check that the input column names, "
            "column order, and data types match "
            "the model's training data."
        )
