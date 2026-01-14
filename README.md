# Breast Cancer Prediction (ML) - Group 8

## Project Overview
This project is a Machine Learning application designed to predict whether a breast tumor is **Malignant** (cancerous) or **Benign** (non-cancerous). It uses the **Breast Cancer Wisconsin Dataset** to analyze clinical features such as tumor radius, texture, and smoothness.

## Team Members (Group 8)
Muhammad Uzair Asghar      2023-uam-2186
Areeb ul Hasan             2023-uam-2187
Yaseen Tahir               2023-uam-2195
Kamran Ali                 2023-uam-2210
Saif Ashraf                2023-uam-2185

## Tech Stack
 **Language:**  Python
 **Libraries:** pandas, scikit-learn, matplotlib, seaborn
 **Dataset:**   Breast Cancer Wisconsin Dataset (Kaggle)

## Installation & Setup
To run this project locally, follow these steps:

1.  **Clone the Repository**
    
    git clone https://github.com/malikuzairasghar/Breast-Cancer-Prediction

2.  **Install Requirements**
    Ensure you have Python installed. Then run:
    
    pip install pandas scikit-learn matplotlib seaborn
    

3.  **Add the Dataset**
    * Download the dataset from Kaggle.
    * Rename the file to `data.csv`.
    * Place it in the project folder.

4.  **Run the Code**
    
    python project.py
    

## Project Results
* **Model Used:** Logistic Regression
* **Evaluation:** The model outputs an accuracy score and generates a Confusion Matrix to visualize performance.
* **Visualizations:**
    * `confusion_matrix.png`:    Shows correct vs. incorrect predictions.
    * `correlation_heatmap.png`: Shows relationships between tumor features.

## Project Structure
* `project.py`: Main source code for training and prediction.
* `data.csv`: The dataset file (not included in repo, must be downloaded).
* `README.md`: Project documentation.
* `Report.pdf`: Detailed project report and methodology.