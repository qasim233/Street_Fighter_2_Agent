import csv
import os
import sys
import time
from controller import Controller, GameState, Command

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
print(f"Data directory path: {DATA_DIR}")

try:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f" Created data directory at: {DATA_DIR}")
    else:
        print(f" Data directory exists at: {DATA_DIR}")
except Exception as e:
    print(f" Failed to create data directory: {e}")
    sys.exit(1)

class DataCollectorBot:
    def __init__(self, player_id, output_file='game_data.csv'):
        self.player_id = player_id
        self.output_file = os.path.join(DATA_DIR, output_file)
        print(f"Output file path: {self.output_file}")
        
        # Test file creation
        try:
            with open(self.output_file, 'a') as f:
                f.write("")  # Test write
            print(f" Can write to file: {self.output_file}")
        except Exception as e:
            print(f"Cannot write to file: {e}")
            sys.exit(1)
            
        self.rows_written = 0
        self.rounds_played = 0
        self.last_round_over = False
        self._init_csv()

    def _init_csv(self):
        header = [
            'player_health', 'opponent_health',
            'player_x', 'player_y', 'opponent_x', 'opponent_y', 'distance',
            'timer', 'has_round_started', 'is_round_over',
            'player_jumping', 'player_crouching', 'player_in_move', 'player_move_id',
            'opponent_jumping', 'opponent_crouching', 'opponent_in_move', 'opponent_move_id',
            'action_left', 'action_right', 'action_up', 'action_down',
            'action_A', 'action_B', 'action_X', 'action_Y',
            'action_L', 'action_R', 'action_select', 'action_start'
        ]
        
        try:
            if not os.path.isfile(self.output_file):
                with open(self.output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                print(f"Created new file with header: {self.output_file}")
            else
                print(f" Using existing file: {self.output_file}")
        except Exception as e:
            print(f" Failed: {e}")
            sys.exit(1)

    def fight(self, state: GameState, player: str) -> Command:
        try:
            if state.is_round_over and not self.last_round_over:
                self.rounds_played += 1
                print(f"\n Round {self.rounds_played} completed!")
                print(f"Total rows collected: {self.rows_written}")

                if self.rounds_played >= 2:
                    print(" Collected 2 rounds — exiting now.")
                    sys.exit(0)

                print(" Waiting for next round:")
                time.sleep(2)

            self.last_round_over = state.is_round_over
            self._log_state(state)

            # Keep game going with movement
            cmd = Command()
            buttons = cmd.player_buttons if player == "1" else cmd.player2_buttons
            buttons.right = state.timer % 60 < 30
            buttons.left = not buttons.right
            return cmd

        except Exception as e:
            print(f"[ ERROR in fight] {e}")
            return Command()

    def _log_state(self, state: GameState):
        try:
            p1 = state.player1
            p2 = state.player2

            def btn_flags(buttons):
                return [int(buttons.up), int(buttons.down), int(buttons.right), int(buttons.left)]

            row = [
                state.timer,
                state.fight_result if state.fight_result is not None else 'NOT_OVER',
                int(state.has_round_started),
                int(state.is_round_over),
                p1.player_id, p1.health, p1.x_coord, p1.y_coord,
                int(p1.is_jumping), int(p1.is_crouching), int(p1.is_player_in_move), p1.move_id,
                *btn_flags(p1.player_buttons),
                p2.player_id, p2.health, p2.x_coord, p2.y_coord,
                int(p2.is_jumping), int(p2.is_crouching), int(p2.is_player_in_move), p2.move_id,
                *btn_flags(p2.player_buttons),
                self.rounds_played + 1
            ]

            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                self.rows_written += 1
                if self.rows_written % 100 == 0:
                    print(f" Logged {self.rows_written} frames")

        except Exception as e:
            print(f"ERROR logging state] {e}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Attempted to write to: {self.output_file}")

if __name__ == '__main__':
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in ('1', '2'):
            print("Usage: python data.py [1|2]")
            sys.exit(1)

        player_arg = sys.argv[1]
        print(f"Data collection for player {player_arg}")
        print(f"Saving to: {os.path.join(DATA_DIR, 'game_data.csv')}\n")
        print("Start Emulator and click Bot.\n")

        bot = DataCollectorBot(player_id=int(player_arg))
        Controller(bot.fight, player_arg).run()

    except KeyboardInterrupt:
        print("\nStopped by user")
        print(f"Rounds played: {bot.rounds_played}, Rows collected: {bot.rows_written}")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        sys.exit(1)
