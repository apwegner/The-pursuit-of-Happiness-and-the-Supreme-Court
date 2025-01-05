# **Predicting Outcomes of Happiness (poH): A Data-Driven Approach**

## **Overview**
This project explores the intersection of legal analysis and data science to predict outcomes related to "Happiness" (poH) in a structured legal context. Using advanced statistical methods and machine learning models, the study aims to identify meaningful relationships between variables and assess their predictive power.

---

## **Key Objectives**
1. **Variable Relationships**:
   - Employ **Cramér's V** to measure the strength of relationships between two categorical variables (ranging from 0 to 1).
   - Recognize that while a higher value (e.g., >0.3) indicates a strong relationship, the directionality of the relationship remains unknown.

2. **Correlation Analysis**:
   - Incorporate **Pearson correlations** for numerical variables to understand directional relationships (-1: negative, 0: none, 1: positive).
   - Address challenges in mixed variable types:
     - Convert non-numerical variables (e.g., True/False) into multi-value representations.
     - Example: A "Right of Privacy" variable was encoded as `1` or `2`, indicating dual usage in specific contexts.
   - **Open Task**: Preserve numerical variables without unnecessary conversions for improved interpretability.

3. **Modeling Approaches**:
   - Develop a model outperforming standard prediction baselines (e.g., linear models).
   - Introduce **Feature Importance** to determine which variables (features) most influence predictions.

---

## **Methodology**
### **1. Feature Importance and Regularization**
- Utilize **decision trees** and **random forests** to understand variable importance.
- Apply **permutation importance** by comparing shuffled and original data to evaluate a feature's predictive power.
- Notable Findings:
  - The **14th Amendment (Amd. XIV)** emerged as a critical feature in distinguishing between "Happiness" and poH outcomes.

### **2. Advanced Correlation Analysis**
- Examine the relationship between feature values and model predictions.
- Account for binary variables by testing multiple configurations (e.g., using either value for test data points).

### **3. Temporal Data Splits**
- Explore **time-based splits** to predict future judgments based on past rulings.
- Investigate scenarios where partial judgments are used as target variables for prediction.

### **4. Model Metrics**
- Evaluate model performance using metrics like **AUROC** (Area Under Receiver Operating Characteristic).

---

## **Challenges and Ongoing Work**
- Implementing meaningful transformations for mixed-value variables without sacrificing interpretability.
- Enhancing the predictive model by exploring:
  - Additional feature combinations.
  - Temporal dependencies in legal data.

---

## **Key Tools and Dependencies**
- **Programming Environment**: Developed primarily using **PyCharm** and **VSCode**.
- **Dependencies**:
   ```commandline
   pip install skrub openpyxl pytabkit sklearn matplotlib scikit-learn autogluon.tabular[all]
   ```

---

## **How to Use**
1. Install required dependencies.
2. Run `run_evaluation.py` to generate prediction models and plots.
3. Examine correlation tables and feature importance outputs to analyze variable relationships and model behavior.

---

## **Acknowledgements**
Special thanks to [David Holzmüller](https://github.com/dholzmueller) for his invaluable assistance with coding and development throughout this project.
