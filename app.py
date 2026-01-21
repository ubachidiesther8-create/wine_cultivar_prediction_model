from flask import Flask, render_template, request
import pandas as pd
import joblib
app = Flask(__name__)
model = joblib.load("model/wine_cultivar_model.pkl")
FEATURES = [
    'alcohol',
    'malic_acid',
    'flavanoids',
    'color_intensity',
    'hue',
    'proline'
]
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            df = pd.DataFrame([input_data], columns=FEATURES)

            pred_class = model.predict(df)[0]

            class_map = {
                0: "Cultivar 1",
                1: "Cultivar 2",
                2: "Cultivar 3"
            }

            prediction = class_map[pred_class]
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction
    )
if __name__ == "__main__":
    app.run(debug=True)
