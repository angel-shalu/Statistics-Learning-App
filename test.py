import streamlit as st
st.set_page_config(layout="wide", page_title="sta2")

st.markdown("""
<style>
.gradient-text{
    text-align: centre;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff4b1f, #1fddff)
    -webkit-background-clip: text;
    -webkit-text-fill-color: tranparent;
}
<style>
<div class="gradient-text> MATHEMATICS STATISTICS APP</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: centre; color:#ff5733;'>STATISTICS</h1>",
    unsafe_allow_html=True 
)

box_type = st.sidebar.selectbox(
    "Types of Statistics",
    ["Descriptive Statistics","Inferential Statistics"]
)

if box_type == "Descriptive Statistics":
    level2_options = [
        "Measures of Centarl Tendency",
        "Measure of Dispersion (Variability)",
        "Measure of Shape",
        "Measures of Position (Relative Standing)"     
    ]
else:
    level2_options = [
        "Hypothesis Testing",
        "Confidence Interval",
        "Z-Test",
        "T-Test",
        "Chi Square Test",
        "Central Limit Theorem"
    ]

box_type1 = st.sidebar.selectbox(
    f"Typeof {box_type},",
    level2_options       
)

if box_type1 == "Measure of Central Tendency":
    level3_options = ["Mean", "Median","Mode"]
    
elif box_type1 == "Measure of Dispersion (Variability)":
    level3_options = [
        "Range", "Mean Absolute Deviation (MAD)",
        "Standard Deviation", "Variance",
        "Coefficient of Variation", "Interquartile Rnage (IQR)"
    ]
    
elif box_type1 == " Measure Of shape":
    level3_options = ["Skewness","Kurtosis"]

elif box_type1 == "Measure of Position (Relative Standing)":
    level3_options = ["Quartiles", "Deciles", "Percentiles", "Z-Score"]

else:
    level3_options = ["Method 1", "Method 2"]

box_type2 = st.sidebar.selectbox(
    f"Select Method {box_type1}",
    level3_options
)

st.markdown(
    f"<h1 style='color:#1f77b4; text-align:center;'>{box_type}</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<h2 style='color:#ff5733;'>{box_type}</h2>",
    unsafe_allow_html=True
)

st.markdown(
    f"<h3 style='color:#2ecc71;'>{box_type}</h3>",
    unsafe_allow_html=True
)

if box_type1.startswith("Measures"):
    st.markdown("### Entry Data (comma-separated)")
    data_input = st.text_input("Example: 10, 20, 30, 40")
    
    if data_input:
        try:
            