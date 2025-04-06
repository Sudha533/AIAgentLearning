import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load API token from .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize Hugging Face InferenceClient
client = InferenceClient(token=HF_API_TOKEN)

app = Flask(__name__)
CORS(app)

# 🔹 Define possible form input types and common mappings
form_input_types = ["text", "number", "email", "date", "checkbox", "select"]

common_field_mappings = {
    "email": "email",
    "dob": "date",
    "birth_date": "date",
    "birthday": "date",
    "birthdate": "date",
    "phone_number": "text",
    "age": "number",
    "phone": "text",
    "zip": "number",
    "postal_code": "text",
    "amount": "number",
    "price": "number",
    "quantity": "number",
    "gender": "select",
    "status": "select",
    "terms_agreed": "checkbox",
    "terms": "checkbox",
    "agreement": "checkbox",
    "preferences": "checkbox",
    "options": "checkbox",
    "choices": "checkbox",
    
    

}

# 🔹 AI-powered function to predict form field type
def predict_field_type(column_name):
    column_name = column_name.lower()

    # Check common mappings first
    if column_name in common_field_mappings:
        return common_field_mappings[column_name]

    try:
        model = "facebook/bart-large-mnli"
        response = client.text_classification(
            inputs=f"What is the best HTML input type for: {column_name}?",
            model=model,
            candidate_labels=form_input_types
        )

        if not response or "label" not in response[0]:
            return "text"  # Default fallback

        predicted_type = response[0]["label"].lower()
        confidence = response[0]["score"]

        # If AI confidence is low, default to "text"
        return predicted_type if confidence > 0.5 else "text"

    except Exception as e:
        print(f"Error in classification: {e}")
        return "text"  # Default fallback

# 🔹 Generate AI-enhanced form schema
@app.route('/form-schema', methods=['GET'])
def get_form_schema():
    file_path = "./assets/test.xlsx"  # Path to your Excel file

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    df = pd.read_excel(file_path)

    form_schema = []
    for column in df.columns:
        field_type = predict_field_type(column)

        # Add extra attributes based on field type
        field_info = {
            "label": column.replace("_", " ").title(),
            "name": column,
            "type": field_type,
            "required": True,
        }

        if field_type == "text":
            field_info["max_length"] = 255
        elif field_type == "number":
            field_info["min"] = 0
            field_info["step"] = 1
        elif field_type == "email":
            field_info["placeholder"] = "Enter your email"
        elif field_type == "date":
            field_info["placeholder"] = "YYYY-MM-DD"

        form_schema.append(field_info)

    return jsonify({"fields": form_schema})

if __name__ == '__main__':
    app.run(debug=True)
