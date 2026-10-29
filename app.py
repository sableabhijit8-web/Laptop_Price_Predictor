import streamlit as st
import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlencode

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="LaptopIQ | Price Predictor",
    page_icon="💻",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #20204a 0%, transparent 35%),
        radial-gradient(circle at 90% 20%, #172c48 0%, transparent 35%),
        #0b1020;
    color: #f5f7ff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(135deg, #24214c, #152d49);
    border: 1px solid rgba(153, 139, 255, 0.3);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 28px;
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.2);
}

.hero-tag {
    color: #b8b1ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-size: clamp(32px, 5vw, 52px);
    font-weight: 800;
    margin: 10px 0;
    color: #ffffff;
}

.hero-description {
    color: #c1c9df;
    font-size: 15px;
    line-height: 1.7;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #ffffff;
    margin: 12px 0 18px 0;
}

div[data-testid="stForm"] {
    background: rgba(21, 29, 53, 0.82);
    border: 1px solid rgba(151, 163, 204, 0.2);
    border-radius: 22px;
    padding: 24px;
}

label, .stMarkdown p {
    color: #dce3f8 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #111a30 !important;
    border: 1px solid #35415f !important;
    border-radius: 10px !important;
    color: white !important;
}

input {
    color: white !important;
}

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7867ff, #536dfe);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #9184ff, #6c82ff);
    color: white;
    border: none;
}

.result-card {
    background: linear-gradient(135deg, #211d43, #142a43);
    border: 1px solid #7867ff;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 12px 35px rgba(50, 45, 120, 0.25);
}

.result-label {
    color: #b8b1ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-price {
    color: white;
    font-size: clamp(32px, 5vw, 46px);
    font-weight: 800;
    margin: 12px 0;
}

.result-caption {
    color: #a6b0ca;
    font-size: 13px;
}

.shop-card {
    background: linear-gradient(135deg, #182d43, #20204a);
    border: 1px solid #465c88;
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #7f8ba9;
    font-size: 12px;
    margin-top: 35px;
}

a[data-testid="stLinkButton"] {
    background: linear-gradient(135deg, #7867ff, #536dfe);
    color: white !important;
    border-radius: 12px;
    font-weight: 700;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("pipe1.pkl", "rb") as file:
        model = pickle.load(file)

    with open("df1.pkl", "rb") as file:
        data = pickle.load(file)

    return model, data


try:
    pipe, df = load_model()

except FileNotFoundError as e:
    st.error(
        f"File not found: {e.filename}. "
        "Keep app.py, pipe1.pkl and df1.pkl in the same folder."
    )
    st.stop()

except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.stop()

# --------------------------------------------------
# VALIDATE DATASET
# --------------------------------------------------
required_columns = [
    "Company",
    "TypeName",
    "Cpu brand",
    "Gpu brand",
    "os"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Missing dataset columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

companies = sorted(df["Company"].dropna().unique().tolist())
laptop_types = sorted(df["TypeName"].dropna().unique().tolist())
cpu_brands = sorted(df["Cpu brand"].dropna().unique().tolist())
gpu_brands = sorted(df["Gpu brand"].dropna().unique().tolist())
operating_systems = sorted(df["os"].dropna().unique().tolist())

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ SMART LAPTOP PRICE ESTIMATOR</div>
    <div class="hero-title">LaptopIQ 💻</div>
    <div class="hero-description">
        Discover an estimated laptop price using machine learning.
        Configure your laptop and explore options within your budget.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.markdown(
    '<div class="section-title">⚙️ Configure Your Laptop</div>',
    unsafe_allow_html=True
)

with st.form("laptop_form"):

    st.markdown("#### 🏷️ Brand and Category")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Laptop company",
            companies
        )

    with col2:
        laptop_type = st.selectbox(
            "Laptop type",
            laptop_types
        )

    st.markdown("#### 🖥️ Display Specifications")

    col1, col2 = st.columns(2)

    with col1:
        resolution = st.selectbox(
            "Screen resolution",
            [
                "1366x768",
                "1600x900",
                "1920x1080",
                "2560x1440",
                "3840x2160"
            ],
            index=2
        )

    with col2:
        screen_size = st.selectbox(
            "Screen size (inches)",
            [11.6, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3],
            index=5
        )

    col1, col2 = st.columns(2)

    with col1:
        touchscreen = st.selectbox(
            "Touchscreen",
            ["No", "Yes"]
        )

    with col2:
        ips = st.selectbox(
            "IPS display",
            ["No", "Yes"]
        )

    st.markdown("#### 🚀 Performance and Storage")

    col1, col2 = st.columns(2)

    with col1:
        ram = st.selectbox(
            "RAM (GB)",
            [2, 4, 6, 8, 12, 16, 24, 32, 64],
            index=3
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=10.0,
            value=1.8,
            step=0.1
        )

    col1, col2 = st.columns(2)

    with col1:
        cpu = st.selectbox(
            "CPU brand",
            cpu_brands
        )

    with col2:
        gpu = st.selectbox(
            "GPU brand",
            gpu_brands
        )

    col1, col2 = st.columns(2)

    with col1:
        hdd = st.selectbox(
            "HDD storage (GB)",
            [0, 128, 256, 512, 1024, 2048],
            index=0
        )

    with col2:
        ssd = st.selectbox(
            "SSD storage (GB)",
            [0, 8, 128, 256, 512, 1024, 2048],
            index=3
        )

    os_choice = st.selectbox(
        "Operating system",
        operating_systems
    )

    predict_clicked = st.form_submit_button(
        "✦ Predict Laptop Price"
    )

# --------------------------------------------------
# PREDICTION AND AMAZON SHOPPING
# --------------------------------------------------
if predict_clicked:

    try:
        touchscreen_value = 1 if touchscreen == "Yes" else 0
        ips_value = 1 if ips == "Yes" else 0

        X_res, Y_res = map(
            int,
            resolution.split("x")
        )

        ppi = (
            (X_res ** 2 + Y_res ** 2) ** 0.5
        ) / screen_size

        # Match training feature names and order.
        # TouchScreen capitalization is important.
        input_data = pd.DataFrame(
            [[
                company,
                laptop_type,
                ram,
                weight,
                touchscreen_value,
                ips_value,
                ppi,
                cpu,
                hdd,
                ssd,
                gpu,
                os_choice
            ]],
            columns=[
                "Company",
                "TypeName",
                "Ram",
                "Weight",
                "TouchScreen",
                "IPS",
                "ppi",
                "Cpu brand",
                "HDD",
                "SSD",
                "Gpu brand",
                "os"
            ]
        )

        prediction = float(pipe.predict(input_data)[0])

        # If training used np.log(Price):
        predicted_price = float(np.exp(prediction))

        # If training used np.log1p(Price), replace the
        # line above with:
        # predicted_price = float(np.expm1(prediction))

        if not np.isfinite(predicted_price) or predicted_price <= 0:
            st.error(
                "The model returned an invalid price. "
                "Check the target transformation used during training."
            )
            st.stop()

        # ------------------------------------------
        # DISPLAY PREDICTED PRICE
        # ------------------------------------------
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    ESTIMATED LAPTOP PRICE
                </div>
                <div class="result-price">
                    ₹{predicted_price:,.0f}
                </div>
                <div class="result-caption">
                    AI-powered estimate · Not a live market quote
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "Actual prices can vary depending on the laptop model, "
            "seller, offers, and availability."
        )

        # ------------------------------------------
        # AMAZON PRICE RANGE
        # ------------------------------------------
        min_price = max(0, int(predicted_price * 0.80))
        max_price = int(predicted_price * 1.20)

        st.markdown(
            """
            <div class="shop-card">
                <h2>🛒 Find Your Laptop on Amazon</h2>
                <p>
                    Browse laptops with similar specifications
                    within your predicted budget.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"**Your shopping range:** "
            f"₹{min_price:,} – ₹{max_price:,}"
        )

        # ------------------------------------------
        # BUILD AMAZON INDIA SEARCH LINK
        # ------------------------------------------
        amazon_query = (
            f"{company} {laptop_type} laptop "
            f"{ram}GB RAM {cpu}"
        )

        params = urlencode({
            "k": amazon_query,
            "low-price": min_price,
            "high-price": max_price
        })

        amazon_url = f"https://www.amazon.in/s?{params}"

        # Clicking opens Amazon India in the browser
        st.link_button(
            "🛍️ Browse & Buy on Amazon",
            amazon_url,
            use_container_width=True
        )

        st.caption(
            "Amazon opens in your browser. Check the exact specifications, "
            "seller, delivery, and final price before purchasing."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.warning(
            "Make sure the feature names, feature order, and preprocessing "
            "match the model's training pipeline."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    LaptopIQ · Machine Learning Laptop Price Predictor
</div>
""", unsafe_allow_html=True)
