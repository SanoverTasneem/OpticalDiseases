from flask import Flask, render_template_string
from data_preprocessing import preprocess_data
from model_pipeline import train_and_evaluate, prepare_data
import pandas as pd

app = Flask(__name__)

# HTML template as a Python string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
</head>
<body>
    <h1>Data Head</h1>
    <pre>{{ data_head }}</pre>
    <h1>Model Evaluation Results</h1>
    <pre>{{ model_results }}</pre>
</body>
</html>
"""

@app.route('/')
def show_results():
    # Path to your dataset
    df_path = 'D:\\Northeastern\\Semester 2\\Python\\Optical Diseases\\full_df.csv'
    df_cleaned = preprocess_data(df_path)

    # Mock-up for displaying purposes - adjust as necessary
    X = df_cleaned[['Patient Age', 'Patient Sex', 'D', 'H']]  # Example feature selection
    y = df_cleaned['MONR']  # Example target variable
    
    # Since train_and_evaluate prints results, consider modifying it to return results as a string
    # For demonstration, let's assume we have a function that returns model results as a string
    model_results = "Model evaluation results placeholder. Modify train_and_evaluate to return actual results."
    
    # Render the HTML page with the data head and model results
    return render_template_string(HTML_TEMPLATE, data_head=df_cleaned.head().to_string(), model_results=model_results)
print("Hello, World!")


if __name__ == '__main__':
    app.run(debug=True)
