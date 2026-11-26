from flask import Flask, render_template, request, jsonify
import joblib
import re

# Step 1: Create Flask application
app = Flask(__name__)

# Step 2: Load the trained ML model
model = joblib.load("model/spam_model.pkl")
def clean_text(message):

    message = message.lower()

    message = re.sub(r"[^a-z0-9\s]", " ", message)

    message = re.sub(r"\s+", " ", message).strip()

    return message

# Step 3: Create the home page
@app.route("/")
def home():
    return render_template("index.html")


# Step 4: Create prediction API
@app.route("/predict", methods=["POST"])
def predict():

    # Receive message from frontend
    data = request.get_json(silent=True) or {}

    message = data.get("message", "")

    # Check empty message
    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "Please enter a valid message."
        }), 400

    # Step 5: Predict using ML model
    cleaned_message = clean_text(message)
    prediction = model.predict([cleaned_message])[0]

    # Step 6: Convert prediction into readable result
    if prediction == "spam":
        result = "SPAM"
    else:
        result = "NOT SPAM"

    # Step 7: Send result to frontend
    return jsonify({
        "prediction": result
    })


# Step 8: Run Flask server
if __name__ == "__main__":
    app.run(debug=True)