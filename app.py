"""
app.py — Flask app serving the prediction form and handling /predict,
matching the original project's server-rendered form design (not a JSON API).

Run with: python app.py
Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("model.pkl not found. Run `python train.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    form = request.form
    try:
        values = {
            "age": float(form["age"]),
            "gender": float(form["gender"]),
            "sleep_hours": float(form["sleep"]),
            "stress_level": float(form["stress"]),
            "social_support": float(form["social"]),
            "physical_activity": float(form["activity"]),
            "work_hours": float(form["work"]),
        }
    except (KeyError, ValueError):
        return render_template("index.html", prediction=None, error="Please fill in every field with a valid number.")

    row = [values[col] for col in feature_columns]
    X = scaler.transform([row])

    pred_label = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    probabilities = {label: round(float(p) * 100, 1) for label, p in zip(model.classes_, probs)}

    return render_template(
        "index.html",
        prediction=pred_label,
        probabilities=probabilities,
        form_values=form,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
