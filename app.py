import streamlit as st
import chess
import chess.svg
import chess.engine
import numpy as np
from PIL import Image
from io import BytesIO
import cairosvg
import xgboost as xgb
import time

# Initialize Stockfish engine (set path to Stockfish binary)
stockfish_path = r"C:\Users\hiddensardar\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({"Threads": 4})  # Use multiple threads for faster analysis

# Load the trained XGBoost model
model_path = r"D:\CAPABLANCA\Models\xgboost_chess_model_28.json"
bst = xgb.Booster()
bst.load_model(model_path)

# Initialize board in session state
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()

# Function to render the chessboard
def render_board(_board):
    board_svg = chess.svg.board(board=_board)
    return board_svg

# Function to convert SVG to image
def svg_to_image(svg_data):
    png_image = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(BytesIO(png_image))
    image = image.resize((400, 400))  # Resize to optimize performance
    return image

# Function to extract features from the board for XGBoost model
def extract_features(board):
    # Example feature extraction: convert board to 64-length feature vector
    # Replace with your actual feature extraction logic
    feature_vector = np.zeros(64)  # Simplified example, replace this
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            feature_vector[square] = piece.piece_type
    return feature_vector

# Function to select the best move from Stockfish's top 10 moves using XGBoost
def select_best_move(board):
    # Get the top 10 moves from Stockfish (multipv=10)
    result = engine.analyse(board, chess.engine.Limit(depth=5), multipv=10)
    
    top_10_moves = [info['pv'][0] for info in result]  # Extract the first move from each of the top 10 lines
    top_10_features = []

    # Display Stockfish's top 10 predicted moves
    st.write("Stockfish's Top 10 Moves:")
    for i, move in enumerate(top_10_moves):
        st.write(f"{i+1}: {board.san(move)}")  # Display moves in SAN format

    # For each move, apply it temporarily, extract features, then undo the move
    for move in top_10_moves:
        board.push(move)  # Make the move on the board
        features = extract_features(board)  # Extract features from the new board state
        top_10_features.append(features)  # Store the features
        board.pop()  # Undo the move to return to the original position

    # Convert the features to a DMatrix for XGBoost
    dmatrix = xgb.DMatrix(np.array(top_10_features))

    # Use XGBoost to predict scores for each move
    predictions = bst.predict(dmatrix)
    
    # Select the move with the highest score
    best_move_index = np.argmax(predictions)
    
    # Display the move selected by the XGBoost model
    st.write(f"Move selected by XGBoost model: {board.san(top_10_moves[best_move_index])}")

    return top_10_moves[best_move_index]  # Return the move selected by the model

# Streamlit app layout
st.title("Play Chess with AI")

# Display the current board (SVG to PNG conversion)
svg_data = render_board(st.session_state.board)
st.image(svg_to_image(svg_data))  # Render the board

# Input for the user's move in SAN format
move = st.text_input("Your Move in SAN (e.g., e4, Nf3):")

# Handle user move submission
if move:
    try:
        # Start timer to profile this section
        start_time = time.time()

        # Make the player's move
        st.session_state.board.push_san(move)

        # Log the updated board after the player's move
        st.write("Player move completed.")

        # AI makes its move using Stockfish + XGBoost
        ai_move_start_time = time.time()
        ai_move = select_best_move(st.session_state.board)
        st.write(f"AI Move processed in {time.time() - ai_move_start_time} seconds")

        # Apply the AI's move to the board
        st.session_state.board.push(ai_move)

        # Re-render the board with both moves
        svg_data = render_board(st.session_state.board)
        st.image(svg_to_image(svg_data))  # Re-render the board

        # End timer
        end_time = time.time()
        st.write(f"Move processed in {end_time - start_time} seconds")

    except Exception as e:
        st.write(f"Invalid move: {e}. Please try again.")

# Check if the game is over
if st.session_state.board.is_game_over():
    st.write("Game Over!")
    result = st.session_state.board.result()
    st.write(f"Result: {result}")

# Close Stockfish engine when done
engine.quit()
