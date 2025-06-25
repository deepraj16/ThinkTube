from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from youtube_rag import initialize_rag_system, query_video, process_video_query
from urllib.parse import urlparse, parse_qs
import numpy as np

app = Flask(__name__)
CORS(app)

# Patch numpy for some libraries if needed
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64

# Global dictionary to store chains per video_id
rag_chains = {}


# Utility to extract video ID from YouTube URL
def extract_video_id(url):
    parsed_url = urlparse(url)
    if 'youtube' in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    if 'youtu.be' in parsed_url.netloc:
        return parsed_url.path.strip('/')
    return None


@app.route("/")
def index():
    return render_template("index3.html")


@app.route("/get_youtube_video_info", methods=["POST"])
def get_youtube_video_info():
    data = request.get_json()
    video_url = data.get("video_id")

    if not video_url:
        return jsonify({"error": "video_id is required"}), 400

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube video URL"}), 400

    video_info = {
        "video_id": video_id,
        "title": "YouTube Video",
        "description": "Video loaded successfully. You can now ask questions about this video.",
        "status": "ready_for_questions"
    }

    return jsonify(video_info)


@app.route("/initialize_video", methods=["POST"])
def initialize_video():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400

    if video_id in rag_chains:
        return jsonify({"status": "success", "message": "Video already initialized."})

    main_chain, status = initialize_rag_system(video_id)
    if main_chain:
        rag_chains[video_id] = main_chain
        return jsonify({"status": "success", "message": "Video initialized."})
    else:
        return jsonify({"error": status}), 400


@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.get_json()
    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id or not question:
        return jsonify({"error": "video_id and question are required"}), 400

    if video_id not in rag_chains:
        main_chain, status = initialize_rag_system(video_id)
        if not main_chain:
            return jsonify({"error": f"Failed to initialize video: {status}"}), 400
        rag_chains[video_id] = main_chain

    main_chain = rag_chains[video_id]
    response = query_video(question, main_chain)

    if response:
        return jsonify({
            "question": question,
            "answer": response,
            "video_id": video_id
        })
    else:
        return jsonify({"error": "Failed to process your question."}), 400


@app.route("/quick_query", methods=["POST"])
def quick_query():
    """Single endpoint for quick questions without separate initialization"""
    data = request.get_json()
    video_id = data.get("video_id")
    question = data.get("question")

    if not video_id or not question:
        return jsonify({"error": "video_id and question are required"}), 400

    result = process_video_query(video_id, question)

    if "error" in result:
        return jsonify(result), 400
    else:
        return jsonify({
            "question": question,
            "answer": result["response"],
            "video_id": video_id
        })


@app.route("/get_current_video", methods=["POST"])
def get_current_video():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400

    is_initialized = video_id in rag_chains
    return jsonify({
        "video_id": video_id,
        "initialized": is_initialized,
        "status": "ready" if is_initialized else "loaded"
    })


if __name__ == "__main__":
    app.run(debug=True)
