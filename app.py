from flask import Flask, jsonify, render_template, request
from analyzer import analyze

app = Flask(__name__)

MAX_LENGTH = 2000  # maximum characters allowed per message


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))[:MAX_LENGTH]
    return jsonify(analyze(message))


if __name__ == "__main__":
    # host 0.0.0.0 lets other devices on the same Wi-Fi open it
    app.run(host="0.0.0.0", port=5000, debug=False)
