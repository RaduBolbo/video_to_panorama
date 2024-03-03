from flask import Flask, request, jsonify
from flask_method import video_to_panorama

app = Flask(__name__)

@app.route('/myapi', methods=['POST'])
def handle_request():
    if not request.is_json:
        return jsonify({"error": "Invalid request format, JSON required."}), 415

    data = request.get_json()
    try:
        result = video_to_panorama(data['videoPath'])
        return jsonify(result)
    except KeyError:
        return jsonify({"error": "Missing videoPath in request."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)














# from flask_method import video_to_panorama
# from flask import Flask, request, jsonify
# app = Flask(__name__)

# @app.route('/myapi', methods=['POST'])
# def handle_request():
#     data = request.json
#     result = video_to_panorama(data['videoPath'])
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)
