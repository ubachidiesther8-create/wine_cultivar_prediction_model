from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the 6-feature logistic regression pipeline
model = joblib.load("wine_cultivar_model.joblib")

# Only 6 features for the app
FEATURES = ['alcohol', 'malic_acid', 'flavanoids', 'color_intensity', 'hue', 'proline']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect inputs and convert to float
            input_data = [float(request.form[feature]) for feature in FEATURES]
            df = pd.DataFrame([input_data], columns=FEATURES)
            pred_class = model.predict(df)[0]
            class_map = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
            prediction = class_map[pred_class]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", features=FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
