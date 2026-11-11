import os
import sys
from flask import Flask, render_template, send_from_directory, jsonify, make_response, request

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def get_video():
    mode = request.args.get("aspect", "")
    if mode == "landscape" or mode == "wide":
        return get_video_landscape()
    elif mode == "portrait" or mode == "tall":
        return get_video_portrait()
    # Default fallback
    video_name = "sakura_landscape.mp4" if os.path.isfile(os.path.join(BASE_DIR, "sakura_landscape.mp4")) else "sakura_3d_botanical_animation.mp4"
    return send_from_directory(BASE_DIR, video_name, mimetype="video/mp4")

@app.route("/video/landscape")
def get_video_landscape():
    video_name = "sakura_landscape.mp4" if os.path.isfile(os.path.join(BASE_DIR, "sakura_landscape.mp4")) else "sakura_3d_botanical_animation.mp4"
    return send_from_directory(BASE_DIR, video_name, mimetype="video/mp4")

@app.route("/video/portrait")
def get_video_portrait():
    video_name = "sakura_portrait.mp4" if os.path.isfile(os.path.join(BASE_DIR, "sakura_portrait.mp4")) else "sakura_3d_botanical_animation.mp4"
    return send_from_directory(BASE_DIR, video_name, mimetype="video/mp4")

@app.route("/preview/<filename>")
def get_preview(filename):
    return send_from_directory(BASE_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sakura Animation Server running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
