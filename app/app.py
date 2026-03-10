from flask import Flask, render_template, request
import numpy as np
import joblib
app=Flask(__name__)

model=joblib.load("../model/depression_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    sleep = float(request.form["sleep"])
    stress = int(request.form["stress"])
    social = int(request.form["social"])
    activity = int(request.form["activity"])
    work = int(request.form["work"])

    data = np.array([[age, gender, sleep, stress, social, activity, work]])

    result = model.predict(data)

    if result[0] == 1:
        prediction = "High Risk of Depression"
    else:
        prediction = "Low Risk"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)