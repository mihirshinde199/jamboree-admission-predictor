# 🎓 Jamboree Education – Graduate Admission Predictor

This project analyzes applicant data to predict the **Chance of Admission** to a graduate program using **Linear Regression**. Built for Jamboree Education, this model can help students evaluate their admission probability based on standardized scores, GPA, SOP/LOR strength, and research experience.

---

## 📦 Dataset Description

The dataset includes the following features:

| Column               | Description                                      |
|----------------------|--------------------------------------------------|
| GRE Score            | GRE Score out of 340                             |
| TOEFL Score          | TOEFL Score out of 120                           |
| University Rating    | University Reputation (1–5)                      |
| SOP                  | Statement of Purpose Strength (1–5)              |
| LOR                  | Letter of Recommendation Strength (1–5)          |
| CGPA                 | Undergraduate GPA (out of 10)                    |
| Research             | Research Experience (0 = No, 1 = Yes)            |
| Chance of Admit      | Target Variable (Range 0–1)                      |

---

## 🧠 Problem Statement

Jamboree wants to estimate the **probability of admission** for students applying to foreign universities based on academic scores, experience, and profile strength. The goal is to:

- Understand key factors influencing admissions.
- Build a predictive model using **Linear Regression**.
- Test regression assumptions (VIF, residuals, homoscedasticity, etc.)
- Evaluate performance using R², RMSE, MAE, etc.

---

## 🧪 Techniques & Tools

- Linear Regression (Statsmodels)
- Ridge & Lasso Regression (Sklearn)
- VIF for Multicollinearity
- Residual Plots, QQ Plot, GQ Test
- MAE, RMSE, R², Adjusted R²
- Python, Pandas, NumPy, Seaborn, Matplotlib, Statsmodels, Sklearn

---

## 📊 Key Steps

1. **EDA**  
   - Distribution & outlier checks
   - Correlation heatmap
   - Scatterplots and pairwise analysis

2. **Data Preprocessing**  
   - Dropped unique ID column  
   - Checked and handled missing values  
   - Outlier visualizations  
   - Feature scaling (after train-test split)

3. **Modeling**  
   - Linear Regression using `statsmodels`
   - Dropped predictors with high p-values
   - Ridge & Lasso to compare regularization

4. **Regression Assumption Testing**  
   - VIF Score (multicollinearity < 5)
   - Residual mean ≈ 0
   - Homoscedasticity test using GQ Test
   - Normal distribution (QQ Plot)
   - Residuals vs Fitted plot

5. **Model Evaluation**  
   - R² and Adjusted R²
   - MAE and RMSE
   - Comparison of training vs testing metrics

---

## 📈 Results Summary

| Metric     | Value       |
|------------|-------------|
| R² Score   | 0.78+       |
| Adj. R²    | 0.76+       |
| MAE        | ~0.03       |
| RMSE       | ~0.05       |

---

## 💡 Actionable Insights

- **CGPA and GRE Score** are the most significant predictors.
- **University Rating**, **LOR**, and **SOP** have positive but secondary influence.
- **Research experience** has noticeable but smaller weight.
- Consider gathering **internship experience, extracurriculars, or essay scores** for better model improvement.

---

## ▶️ How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/jamboree-admission-predictor.git

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run main script
python main.py
