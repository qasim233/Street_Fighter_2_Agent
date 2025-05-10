from command import Command
from buttons import Buttons
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import argparse

class Bot:
    def __init__(self, model_type='nn'):
        """
        Initialize bot with specified model type
        :param model_type: 'nn' for neural network or 'xgb' for XGBoost
        """
        self.model_type = model_type
        self.model = None
        self.scaler = joblib.load('model/scaler.pkl')
        columns = np.load('model/columns.npz', allow_pickle=True)
        self.feature_columns = columns['feature_columns']
        self.action_columns = columns['action_columns']
        self.command = Command()
        
        
        if model_type == 'nn':
            self.model = load_model('model/sf2_model.h5')
        elif model_type == 'xgb':
            xgb_path = 'model/xgboost_model.pkl'
            if os.path.exists(xgb_path):
                self.model = joblib.load(xgb_path)
            else:
                raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'nn' or 'xgb'")

    def fight(self, game_state, player):
        # Select which player's perspective to use
        if player == "1":
            me = game_state.player1
            opponent = game_state.player2
        else:
            me = game_state.player2
            opponent = game_state.player1

        # Extract features
        features = self.extract_features(me, opponent, game_state)

        # Predict actions
        X = self.scaler.transform([features])
        
        if self.model_type == 'nn':
            y_pred = self.model.predict(X)[0]
        else:  # XGBoost
            y_pred = self.model.predict_proba(X)
            # Extract positive class probabilities (second column for each action)
            y_pred = np.array([pred[0][1] for pred in y_pred])
        
        y_binary = (y_pred > 0.3).astype(int)
        print(f"Predicted actions ({self.model_type}): {dict(zip(self.action_columns, y_binary))}")

        # Map prediction to buttons
        buttons = Buttons()
        for i, action in enumerate(self.action_columns):
            if y_binary[i]:
                setattr(buttons, action.replace("action_", ""), True)

        # Return command
        if player == "1":
            self.command.player_buttons = buttons
        else:
            self.command.player2_buttons = buttons

        return self.command

    def extract_features(self, me, opponent, game_state):
        distance = abs(me.x_coord - opponent.x_coord)
        return [
            me.health,
            opponent.health,
            me.x_coord,
            me.y_coord,
            opponent.x_coord,
            opponent.y_coord,
            distance,
            game_state.timer,
            int(game_state.has_round_started),
            int(game_state.is_round_over),
            int(me.is_jumping),
            int(me.is_crouching),
            int(me.is_player_in_move),
            me.move_id,
            int(opponent.is_jumping),
            int(opponent.is_crouching),
            int(opponent.is_player_in_move),
            opponent.move_id
        ]

def parse_args():
    parser = argparse.ArgumentParser(description='Street Fighter AI Bot')
    parser.add_argument('--model', type=str, default='nn',
                       choices=['nn', 'xgb'],
                       help='Model type to use (nn: neural network, xgb: XGBoost)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    bot = Bot(model_type=args.model)