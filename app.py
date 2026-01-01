import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Statistics Learning App", layout="wide")

# ---------- Header ----------
st.markdown("<h1 style='text-align:center;color:#4B8BFF;'>📊 Statistics Learning App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Learn statistics through interaction & visualization</p>", unsafe_allow_html=True)
st.divider()

st.markdown("""
<style>
.gradient-tect{
    text-align: centre;
    font-size: 48px;
    font-weight; bold;
    background: linear-gradient(90deg, #ff4b1f, #1fddff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
<style>
<div class="gradient-text"> MATHEMATICS STATICS APP</div>
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

if box_type1 == "Measure of Dispersion (Variability)":
    level3_options = [
        "Range", "Mean Absolute Deviation (MAD)",
        "Standard Deviation", "Variance",
        "Coefficient of Variation", "Interquartile Rnage (IQR)"
    ]
    
elif box_type1 == "Measure of Central Tendency":
    level3_options = ["Mean", "Median","Mode"]
    
elif box_type1 == " Measure Of shape":
    level3_options = ["Skewness","Kurtosis"]

elif box_type1 == "Measure of Position (Relative Standing)":
    level3_options = ["Quartiles", "Deciles", "Percentiles", "Z-Score"]

elif box_type1 == "Hypothesis Testing":
    level3_options = ["Type 1 Error", "Type 2 Error"]

elif box_type1 == "Z-Test" or box_type1 == "T-Test":
    level3_options = ["Population", "Sample"]

elif box_type1 == "Chi Square Test":
    level3_options = ["Goodness of Fit", "Test of Independence"] 

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

# =======================
# CALCULATION SECTION
# =======================

st.markdown("## Data Input Method")
if "data_input_value" not in st.session_state:
    st.session_state.data_input_values = ""

if "inferential_input_value" not in st.session_state:
    st.session_state.inferential_input_value = ""

data_source = st.radio(
    "Choose data input type:",
    ["Manual Entry", "Upload CSV / Excel"],
    horizontal=True
)

data = None

if data_source == "Upload CSV / Excel":
    uploaded_file = st.file_uploader(
        "Upload CSV OR Excel file",
        type=["csv","xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df.pd.read_csv(uploaded_file)
            else:
                df.pd.read_excel(uploaded_file)

            st.success("File uploaded successfully!")
            st.dataframe(df)

            numeric_cols = df.select_dtype(include="number").columns.tolist()

            if not numeric_cols:
                st.error("No numeric column found in file")
            else:
                selected_col = st.selectbox(
                    "Select Numeric Colun for Analysis",
                    numeric_cols
                )

                data = df[selected_col].dropna().astype(float).tolist()
                csv_values = ", ".join(str(x) for x in data)

                st.session_state.data_input_value = csv_values
                st.session_state.inferential_input_value = csv_values

        except Exception as e:
            st.error(f"Error reading file:{e}")

if box_type1.startswith("Measures"):

    if data_source == "Manual Entry":
        st.markdown("### Entry Data (comma-separated)")

    data_input = st.text_input(
        "Example: 10, 20, 30, 40",
        value=st.session_state.data_input_values
    )

    if data is not None and len(data) > 1:
        try:
            data = [float(x.strip()) for x in data_input.split(",")]
        except ValueError:
            st.error("Please entre valid numeric values only.")

    if data_input:
        try:
            data = [float(x.strip()) for x in data_input.split(", ")]
            n = len(data)
            data_stored = sorted(data)
            mean = sum(data) / n
            st.markdown("### Formula & Calculation")

            # ----------CENTRAL TENDENCY -------
            if box_type2 =="Mean":
                st.latex(r"\bar{x}= \frac{\sum x}{n}")
                st.success(f"Mean ={mean}")

            elif box_type2 == "Median":
                st.latex(r"\text{Median} = \text{Median value}")
                if n % 2 == 0:
                    median = (data_stored[n//2 - 1]+ data_stored[n//2])/2
                else:
                    median = data_stored[n//2]
                st.success(f"Median = {median}")

            elif box_type2 == "Mode":
                from collections import Counter
                st.latex(r"\text{Mode} = \text{Mode frequent value}")
                freq = Counter(data)
                max_freq = max(freq.values())
                mode = [k for k, v in freq.items() if v == max_]
                st.success(f"Mode = {mode}")

            # ------------DISPERSION-----------
            elif box_type2 == "Range":
                st.latex(r"\text{Range} = \max(x)-\min(x)")
                st.success(f"Range = {max(data)-min(data)}")

            elif box_type2 == "Mean Absolute Deviation (MAD)":
                st.latex(r"\text{MAD} = \FRAC{\sum|x - \bar{x}|}{n}")
                mad = sum(abs(x-mean) for x in data) / n
                st.success(f"MAD = {mad}")

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
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                st.success(f"IQR = {iqr}")

            elif box_type2 == "Coefficient of Variation (CV)":
                st.latex(r"\text{CV} = \frac{\sigma}{\bar{x}} \times 100")
                std_dev = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
                cv = (std_dev / mean) * 100
                st.success(f"Coefficient of Variation = {cv}%")

            # -----------SHAPE-----
            elif box_type2 == "Skewness":
                st.latex(r"\text{Skewness} = \frac{\frac{1}{n}\sum (x - \bar{x})^3}{\left(\frac{1}{n}\sum (x - \bar{x})^2\right)^{3/2}}")
                skew = sum((x - mean) ** 3 for x in data) / n
                skew = skew / ((sum((x - mean) ** 2 for x in data) / n) ** 1.5)
                st.success(f"Skewness = {skew}")

            elif box_type2 == "Kurtosis":
                st.latex(r"\text{Kurtosis} = \frac{\frac{1}{n}\sum (x - \bar{x})^4}{\left(\frac{1}{n}\sum (x - \bar{x})^2\right)^2}")
                kurt = sum((x - mean) ** 4 for x in data) / n
                kurt = kurt / ((sum((x - mean) ** 2 for x in data) / n) ** 2)
                st.success(f"Kurtosis = {kurt}")

            # ------------POSITION-----------
            elif box_type2 == "Quartiles":
                st.latex(r"Q_k = \text{Value at } \frac{k(n+1)}{4}\text{th position}")
                q1 = np.percentile(data, 25)
                q2 = np.percentile(data, 50)
                q3 = np.percentile(data, 75)
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
            
            
    
# ----------GRAPH SECTION ----------
if data is not None and len(data) > 1:
    st.markdown("### Data Visualization (Seaborn)")
    df_plot = pd.DataFrame({"Values":data})
    
    # Histogram + KDE
    fig1, ax1 = plt.subplots()
    sns.histplot(
        df_plot["Values"],
        bins=10,
        kde=True,
        ax=ax1
    )
    ax1.set_title("Histogram with KDE")
    ax1.set_xlabel("Values")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    
    # Boxplot 
    fig2, ax2= plt.subplots()
    sns.boxplot(
        x=df_plot["Values"],
        ax=ax2
    )
    ax2.set_title("Boxplot")
    ax2.set_xlabel("Values")
    st.pyplot(fig2)
    
# ----------INFERENTIAL STATISTICS SECTION---------

if box_type == "Inferential Statistics":
    st.markdown("### Enter Sample Data(comma-separated)")
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
            
            # ----Hypothesis testing -----------
            if box_type1 == "Hypothesis Testing" and box_type2 == "Type I Error":
                
                st.markdown("### Type I Error")
                st.latex(r"\alpha= P(\text{Reject } H_0 \text{ is true})")
                
                alpha = st.selectbox(
                "Select Significance Level",
                [0.10, 0.05, 0.01]
                )
                
                st.success(f"Type 1 Error = {alpha}")
                st.info("Note: Type I Error is NOT calculated, it is pre-defined.")
                
            elif box_type1 == "Hypothesis Testing" and box_type2 == "Type II Error":
                st.markdown("### Type II Error")
                st.latex(r"beta = P(\test{Fail to reject}H_0\mid H_1 \text{ is true})")
                
                st.markdown("### Inputs")
                mu0 = st.number_input("Null Hypothesis Mean", value=50.0)
                mu1 = st.number_input("Actual Mean", value=55.0)
                sigma = st.number_input("Population Std Dev", value=10.0)
                n = st.number_input("Sample Size (n)", value=25)
                alpha = st.selectbox("Significance Level", [0.10, 0.05, 0.01])
                
                #  Critical Z value 
                z_critical = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}[alpha]
                SE = sigma / math.sqrt(n)
                lower = mu0 - z_critical * SE
                upper = mu0 + z_critical * SE
                z1 = (lower - mu1) / SE
                z2 = (upper - mu1) / SE
                
                from math import erf, sqrt
                
                def normal_cdf(z):
                    return 0.5 * (1 + erf(z / sqrt(2)))
                
                beta = normal_cdf(z2) - normal_cdf(z1)
                power = 1 - beta
                
                st.markdown("### Formula used")
                st.latex(r"\beta = P (L< \bar{x} < U \mid \mu = \mu_1)") 
                st.success(f"Type II Eroor = {beta:.4f}")
                st.success(f"Power of Test = {power:.4f}")
                
            # ---------Z-TEST------------
            if box_type1 == "Z-Test":
                st.markdown("### Z-Test(One Sample)")
                st.latex(r"z = \ frsc{\bar{x} - \mu}{sigma / \sqrt{n}}")
                
                # DEFINE FIRTS (IMPORTANT)
                mu = st.number_input("Hypoyhesized Mean")
                
                if box_type2 =="Population":
                    st.info("Population Z-Test")
                    
                    sigma = st.number_input(
                     "Population Std Dev",
                    value=std_dev
                    )
                    
                    z= (mean - mu) / (sigma / math.sqrt(n))
                    st.success(f"Z-Statistics (Population) ={z}")
                    
                elif box_type2 == "Sample":
                    st.info("Sample Z-Test ( mu estimated feom sample)")
                    
                    s = std_dev
                    z = (mean -mu) / (s / math.sqrt(n))
                    st.success(f"Z-Statisics (Sample) = {z}")
                    
                    x_vals = [i / 10 for i in range(-50, 51)]
                    y_vals = [math.exp(-0.5 * x*x0) / math.sqrt(2*math.pi) for x in x_vals]
                    
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

                if box_type2 == "Population":
                    st.warning(
                    "⚠️ T-Test is generally used when population σ is UNKNOWN.\n"
                    "Population option is not recommended."
                    )

                elif box_type2 == "Sample":
                    st.info("📌 Sample T-Test (σ unknown)")

                    t = (mean - mu) / (std_dev / math.sqrt(n))
                    st.success(f"✅ T-Statistic (Sample) = {t}")

        # Graph (approx t distribution)
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
                
                # ==========Chi Square Test===========
            elif box_type1 == "Chi Square Test":

                st.markdown("### 📘 Chi-Square Test")
                st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
                
                if uploaded_file is not None:
                    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                    
                    if not cat_cols:
                        st.error("❌ No categorical column found in CSV")
                    else:
                        cat_col = st.selectbox(
                           "Select Categorical Column",
                            cat_cols
                        )
                        
                    # ========GOODNESS OF FIT=======
                if box_type2 == "Goodness of Fit":
                    st.info("📌 Chi-Square Goodness of Fit Test (CSV Based)")
                    obs_input = st.text_input(
                        "Enter Observed Frequencies (comma-separated)",
                        "20, 30, 50"
                        )
                    
                    exp_input = st.text_input(
                        "Enter Expected Frequencies (comma-separated)",
                        "25, 25, 50"
                    )
                    
                    if obs_input and exp_input:
                        try:
                            observed = [float(x.strip()) for x in obs_input.split(",")]
                            expected = [float(x.strip()) for x in exp_input.split(",")]
                            
                            if len(observed) != len(expected):
                                st.error("❌ Observed and Expected must have same length")
                            else:
                                chi_square = sum(
                                     (o - e) ** 2 / e for o, e in zip(observed, expected)
                                )
                                
                                df = len(observed) - 1
                                st.success(f"✅ Chi-Square Statistic = {chi_square:.4f}")
                                st.info(f"🎯 Degrees of Freedom = {df}")
                                
                                df_plot = pd.DataFrame({
                                    "Observed": observed,
                                    "Expected": expected
                                })
                                
                                fig, ax = plt.subplots()
                                df_plot.plot(kind="bar", ax=ax)
                                ax.set_title("Observed vs Expected Frequencies")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                                
                        except ValueError:
                            st.error("❌ Please enter valid numeric values")
                            
                    elif box_type2 == "Test of Independence":
                           st.info("🎯 Chi-Square Test of Independence (2×2 Table)")
                           st.markdown("### Enter Contingency Table Values") 
                           
                           col1, col2 = st.columns(2)
                           with col1:
                                a = st.number_input("Cell A", min_value=0, value=10)
                                b = st.number_input("Cell B", min_value=0, value=20)
                               
                           with col2:
                                c = st.number_input("Cell C", min_value=0, value=30)
                                d = st.number_input("Cell D", min_value=0, value=40)

                                observed = [[a, b], [c, d]]

                                row_totals = [sum(row) for row in observed]
                                col_totals = [a + c, b + d]
                                total = sum(row_totals)
   
                                expected = [
                                    [(row_totals[i] * col_totals[j]) / total for j in range(2)]
                                    for i in range(2)
                                ]

                                chi_square = 0
                                for i in range(2):
                                    for j in range(2):
                                        chi_square += (observed[i][j] - expected[i][j]) ** 2 / expected[i][j]

                                df = (2 - 1) * (2 - 1)

                                st.success(f"✅ Chi-Square Statistic = {chi_square:.4f}")
                                st.info(f"🎯 Degrees of Freedom = {df}") 
                                
                                df_table = pd.DataFrame(
                                    observed,
                                    columns=["Column 1", "Column 2"],
                                    index=["Row 1", "Row 2"]
                                )

                                st.table(df_table)

        except ValueError:
            st.error("❌ Please enter valid numeric values only.")
    