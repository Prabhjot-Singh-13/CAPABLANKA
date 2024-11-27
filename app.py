import os
from flask import Flask, render_template, request, jsonify
import chess
import chess.engine
import numpy as np
import pickle
import xgboost as xgb

# Initialize Flask app
app = Flask(__name__)

# Dynamic Stockfish Path
STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), "stockfish.exe")

# Load Models
piece_model = xgb.Booster()
piece_model.load_model(os.path.join(os.path.dirname(__file__), "models/50_player_code.json"))
time_model = xgb.Booster()
time_model.load_model(os.path.join(os.path.dirname(__file__), "models/time_v2.json"))

with open(os.path.join(os.path.dirname(__file__), "models/move_encoder.pkl"), "rb") as f:
    move_encoder = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "models/scaler_X.pkl"), "rb") as f:
    scaler_X = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), "models/scaler_y.pkl"), "rb") as f:
    scaler_y = pickle.load(f)

# Initialize Stockfish Engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/play", methods=["POST"])
def play():
    player_name = request.form.get("player")
    game_type = request.form.get("game")
    return render_template("play.html", player_name=player_name, game_type=game_type)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    board_state = data.get("board_state")
    player_id = data.get("player_id")
    game_phase = data.get("game_phase")

    # Convert board_state to features
    board_features = board_to_features(board_state)
    dmatrix = xgb.DMatrix(np.array([board_features]))

    # Predict best move
    predictions = piece_model.predict(dmatrix)
    best_move_index = np.argmax(predictions)
    best_move = move_encoder.inverse_transform([best_move_index])[0]

    return jsonify({"move": best_move})

@app.route("/players")
def get_players():
    player_type = request.args.get("type")
    players = []

    if player_type == "top30":
        players = [
            "Magnus Carlsen", "Hikaru Nakamura", "Fabiano Caruana",
            "Alireza Firouzja", "Ian Nepomniachtchi", "Wesley So",
            # Add all top 30 players
        ]
    elif player_type == "top20women":
        players = [
            "Hou Yifan", "Ju Wenjun", "Koneru Humpy", "Kateryna Lagno",
            "Harika Dronavalli", "Aleksandra Goryachkina",
            # Add all top 20 women players
        ]

    return jsonify({"players": players})


def board_to_features(fen):
    # Function to convert FEN (chess board state) to features
    piece_mapping = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, '.': 0}
    board = chess.Board(fen)
    return [piece_mapping[str(piece)] if piece else 0 for piece in board.piece_map().values()]

if __name__ == "__main__":
    app.run(debug=True)
