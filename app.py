import csv
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

from api import SentimentAnalysis


PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "template"))
sentiment_analysis = SentimentAnalysis()
FEEDBACK_PATH = Path(app.instance_path) / "feedback.csv"


def append_feedback(text: str, sentiment: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0

    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["text", "sentiment"])
        writer.writerow([text, sentiment])


@app.route("/", methods=["GET"])
def home():
    size = sentiment_analysis.get_size()
    return render_template(
        "index.html", negative_size=size["negative"], positive_size=size["positive"]
    )


@app.route("/result", methods=["POST"])
def result():
    text = request.form.get("text", "").strip()
    if not text:
        abort(400, description="Text is required.")

    prediction, negative_score, positive_score = sentiment_analysis.predict(text)
    return render_template(
        "result.html",
        text=text,
        result=prediction,
        result_negative=negative_score,
        result_positive=positive_score,
    )


def save_feedback(sentiment: str):
    text = request.form.get("text", "").strip()
    if not text:
        abort(400, description="Text is required.")

    append_feedback(text, sentiment)
    return redirect(url_for("home"))


@app.route("/add_to_negative", methods=["POST"])
def add_to_negative():
    return save_feedback("negative")


@app.route("/add_to_positive", methods=["POST"])
def add_to_positive():
    return save_feedback("positive")


if __name__ == "__main__":
    app.run(debug=False)
