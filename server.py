import os

from flask import Flask, request, jsonify

from skin_tone_analyzer import analyze_face_tone

app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>Skin Tone Analyzer</h1>"


@app.route('/fetch_skin_tone', methods=['POST'])
def fetch_skin_tone():
    if request.method == 'POST':
        f = request.files['image']
        filename = f.filename
        f.save(filename)
        result = analyze_face_tone(filename)
        print(result)
        os.remove(filename)
        return jsonify(result)


app.run(host='0.0.0.0')
