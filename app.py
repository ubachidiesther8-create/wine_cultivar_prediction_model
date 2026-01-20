from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained Logistic Regression pipeline
model = joblib.load("wine_logreg_pipeline.joblib")

# Feature names in the correct order
FEATURES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
    'proline'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get user inputs from form
            input_data = [float(request.form[feature]) for feature in FEATURES]
            
            # Convert to DataFrame
            df = pd.DataFrame([input_data], columns=FEATURES)
            
            # Predict class
            pred_class = model.predict(df)[0]
            
            # Map class to label (optional)
            class_map = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
            prediction = class_map[pred_class]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", features=FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
