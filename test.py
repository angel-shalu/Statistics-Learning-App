import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Statistics Learning App", layout="wide")

# ---------- Header ----------
st.markdown("""
<style>
.gradient-text {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff4b1f, #1fddff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}   
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-top: 8px;
}
</style> 

<div class="gradient-text">MATHEMATICS STATICS APP</div>
<div class="subtitle">Learn statistics through interaction & visualization</div>
""", unsafe_allow_html=True)
st.divider()
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
            data = [float(x.strip()) for x in data_input.split(",")]
            n = len(data)
            data_sorted = sorted(data)
            mean = sum(data) / n
            
            st.markdown("### Formula & Calculation")
            
            # ----------- CENTRAL TENDENCY ---------------
            if box_type2 == "Mean":
                st.latex(r"\bar{x}=\frac{\sum x}{n}")
                st.success(f"Mean = {mean}")
                
            elif box_type2 == "Median":
                st.latex(r"\text{Median}=\text{Median value}")
                if n % 2 == 0:
                    median = (data_sorted[n//2 - 1] + data_sorted[n//2]) / 2
                else:
                    median = data_sorted[n//2]
                st.success(f"Median = {median}")
                
            elif box_type2 == "Mode":
                from collections import Counter 
                st.latex(r"\text{Mode} = \text{Most frequent value}")
                freq = Counter(data)
                max_freq = max(freq.values())
                mode = [k for k, v in freq.items() if v == max_freq]
                st.success(f"Mode = {mode}")
                
            #----------DISPERSION----------
            elif box_type == "Range":
                st.latex(r"\text{Range} = \max(x) - \min(x)")
                st.success(f"Range = {max(data) - min(data)}")   
                
            elif box_type2 == "Mean Absolute Deviation (MAD)":
                st.latex(r"\tet{MAD} = \frac{\sum |x- \bar{x}|}{n}")
                mad = sum (abs(x-mean) for x in data) / n
                st.success(f"MAD =N{mad}")
                
            
            elif box_type2 == "Vriance":
                st.latex(r"\sigma^2 = \frac{\sum |x - \bar{x}^2}{n}")
                variance = sum((x - mean) ** 2 for x in data) / n
                st.success(f"Variance = {variance}")

            elif box_type2 == "Standard Deviation":
                st.latex(r"\sigma = \sqrt{frac{\sum (x - \bar{x})^2}{n}")
                std_dev = sum((x - mean) ** 2 for x in data) / n**0.5
                st.success(f"Variance = {std_dev}")

            elif box_type2 == "Interquartile Range (IQR)":
                st.latex(r"\text{IQR} = Q_3 - Q_1")
                q1 = data_stored[n//4]
                q3 = data_stored[(3*n)//4]
                st.success(f"IQR = {q3 -  q1}")

            elif box_type2 == "Coefficient of Variation (CV)":
                st.latex(r"\text{CV} = \frac{\sigma}{\bar{x}} \times 100")
                std_dev = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
                cv = (std_dev / mean) * 100
                st.success(f"Coefficient of Variation = {cv:.2f}%")

            # -----------SHAPE-----
            elif box_type2 == "Skewness":
                st.latex(r"\text{Skewness} = \frac{\sum (x - \bar{x})^3}{n\sigma^3}")
                std_dev = (sum((x - mean) ** 3 for x in data) / n) ** 0.5
                skew = sum((x - mean) ** 2 for x in data) / (n * std_dev ** 3)
                st.success(f"Skewness = {skew}")

            elif box_type2 == "Kurtosis":
                st.latex(r"\text{Kurtosis} = \frac{\sum(x - \bar{x})^3}{n\sigma^3}")
                std_dev = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
                kurt = sum((x - mean) ** 4 for x in data) / (n * std_dev ** 4)
                st.success(f"Kurtosis = {kurt}")

            # ------------POSITION-----------
            elif box_type2 == "Quartiles":
                st.latex(r"Q_1, Q_2, Q_3")
                q1 = data_sorted[n//4]
                q2 = data_sorted[n//2]
                q3 = data_sorted[3*n//4]
                st.success(f"Q1 = {q1}, Q2 = {q2}, Q3 = {q3}")

            elif box_type2 == "Deciles":
                st.latex(r"D_k = \frac{k(n+1)}{10}}")
                decile = {f"D{k}":data_stored[int(k*n/10)-1] for k in range(1,10)}
                st.success(f"Deciles = {decile}")

            elif box_type2 == "Percentile":
                st.latex(r"P_k =  \frac{k(n+1)}{100}")
                p = st.slider("Select Percentile", 1,99)
                index = int(p*n/100)
                st.success(f"P{p} = {data_stored[index]}")

            elif box_type2 == "Z-Score":
                st.latex(r"Z = \frac{x - \bar{x}}{\sigma}")
                value = st.number_input("Enter value of Z-score")
                std_dev = (sum ((x-mean) **2 for x in data) / n) ** 0.5    
                z = (value - mean) / std_dev
                st.success(f"Z-Scores = {z}")

        except ValueError:
            st.error("Please enter the valid numeric value only")
            
if box_type1.startswith("Measures") and data_input:
    st.markdown("### Data Visualization")
    
    # Histogram
    fig1, ax1 = plt.subplots()
    ax1.hist(data, bins=10)
    ax1.set_title("Histogram")
    ax1.set_xlabel("Values")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    
    # Boxplot
    fig2, ax2 = plt.subplots()
    ax2.boxplot(data, ver=False)
    ax2.set_title("Histogram")
    ax2.set_xlabel("Values")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    
# ============ INFERENTIAL STATISTICS ==============
 
import math
import matplotlib.pyplot as plt

if box_type2 == "Inferential Statistics":
    st.markdown("### Enter Sample Data")
    data_input_inf = st.text_input(
        "Example: 10, 12, 14, 13, 12",
        value = st.session_state.inferential_input_value,
        key="inferential"
    )
    
    if data_input_inf:
        try:
            data = [float(x.strip()) for x in data_input_inf.split(",")]
            n = len(data)
            mean = sum(data) / n
            variance = sum((x-mean) ** 2 for x in data) / (n-1)
            std_dev = math.sqrt(variance)
            
            if box_type1 == "Z-Test":
                st.markdown("### Z-Test(One Sample)")
                st.latex(r"z = \ frsc{\bar{x} - \mu}{sigma / \sqrt{n}}")
                
                # DEFINE FIRTS (IMPORTANT)
                mu = st.number_input("Population Mean")
                sigma = st.number_input("Population Std Dev", value=std_dev)
                z= (mean - mu) / (sigma / math.sqrt(n))
                st.success(f"Z-Statistics ={z}")
                    
                x_vals = [i / 10 for i in range(-50, 51)]
                y_vals = [math.exp(-0.5 * x*x) / math.sqrt(2*math.pi) for x in x_vals]
                    
                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals)
                ax.axvline(z)
                ax.set_title("Standard Normal Distribution (Z-Test)")
                st.pyplot(fig)
                    
            # ============ T- TEST ===========
            elif box_type1 == "T-Test":

                st.markdown("### 📘 T-Test (One Sample)")
                st.latex(r"t = \frac{\bar{x} - \mu}{s / \sqrt{n}}")

                mu = st.number_input("Hypothesized Mean (μ)")
                t = (mean - mu) / (std_dev / math.sqrt(n))
                st.success(f"T-Statistics = {t}")
                
                x_vals = [i / 10 for i in range(-50, 51)]
                y_vals = [math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi) for x in x_vals]

                fig, ax = plt.subplots()
                ax.plot(x_vals, y_vals)
                ax.axvline(t)
                ax.set_title("Sampling Distribution (T-Test Approx)")
                st.pyplot(fig)
                
            elif box_type1 == "Confidence Interval":

                st.markdown("### 📘 Confidence Interval for Mean")
                st.latex(r"\bar{x} \pm z \cdot \frac{\sigma}{\sqrt{n}}")

                confidence = st.selectbox(
                    "Confidence Level",
                    [90, 95, 99]
                )

                z_values = {90: 1.645, 95: 1.96, 99: 2.576}
                z = z_values[confidence]

                margin = z * (std_dev / math.sqrt(n))
                lower = mean - margin
                upper = mean + margin

                st.success(
                    f"✅ {confidence}% CI = ({lower:.2f}, {upper:.2f})"
                )

                # Graph
                fig, ax = plt.subplots()
                ax.errorbar(mean, 0, xerr=margin, fmt='o')
                ax.set_title("Confidence Interval")
                ax.set_yticks([])
                st.pyplot(fig)
                
                # ========CLT===========
            elif box_type1 == "Central Limit Theorem":
                st.markdown("### 📘 Central Limit Theorem")
                st.latex(r"\bar{X} \sim N(\mu, \sigma / \sqrt{n})")

                fig, ax = plt.subplots()
                ax.hist(data, bins=10)
                ax.set_title("Sample Distribution (CLT Illustration)")
                st.pyplot(fig)
             
        except ValueError:
            st.error("❌ Please enter valid numeric values")   
                    
    