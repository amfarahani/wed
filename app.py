from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "file_iris1.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template(
        "index.html",
        predict="",
        values={}
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {
            "sepal_length": request.form.get("sepal_length", ""),
            "sepal_width": request.form.get("sepal_width", ""),
            "petal_length": request.form.get("petal_length", ""),
            "petal_width": request.form.get("petal_width", "")
        }

        X = np.array([[float(values["sepal_length"]),
                       float(values["sepal_width"]),
                       float(values["petal_length"]),
                       float(values["petal_width"])]], dtype=float)

        pred = model.predict(X)[0]

        return render_template(
            "index.html",
            predict=str(pred),
            values=values
        )

    except Exception:
        return render_template(
            "index.html",
            predict="",
            values={}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


