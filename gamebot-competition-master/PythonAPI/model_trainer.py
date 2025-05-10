import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > 0.6:
            print("Stopping training as accuracy exceeded 60%")
            self.model.stop_training = True

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, test_size=0.2):
    """
    Load and preprocess the Street Fighter II dataset
    """
    print(f"Loading data from {file_path}...")
    
    # Try to detect the delimiter automatically
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        if '\t' in first_line:
            delimiter = '\t'
            print("Detected tab delimiter")
        else:
            delimiter = ','
            print("Detected comma delimiter")
    
    # Load the data with the detected delimiter
    df = pd.read_csv(file_path, delimiter=delimiter)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select relevant features for the model
    feature_columns = [
        'player_health', 'opponent_health',
        'player_x', 'player_y', 'opponent_x', 'opponent_y', 'distance',
        'timer', 'has_round_started', 'is_round_over',
        'player_jumping', 'player_crouching', 'player_in_move', 'player_move_id',
        'opponent_jumping', 'opponent_crouching', 'opponent_in_move', 'opponent_move_id'
    ]
    
    # Select action columns (outputs)
    action_columns = [
        'action_left', 'action_right', 'action_up', 'action_down',
        'action_A', 'action_B', 'action_X', 'action_Y',
        'action_L', 'action_R'#, 'action_select', 'action_start'
    ]
    
    # Check if all columns exist in the dataframe
    missing_columns = []
    for col in feature_columns + action_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        print(f"Warning: {len(missing_columns)} columns not found in dataset: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        raise ValueError("Required columns missing from dataset")
    
    # Filter out rows where the round hasn't started or is over
    df_filtered = df[df['has_round_started'] == 1]
    df_filtered = df_filtered[df_filtered['is_round_over'] == 0]
    
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Extract features and labels
    X = df_filtered[feature_columns].values
    y = df_filtered[action_columns].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns, action_columns

def build_model(input_dim, output_dim):
    """
    Build a neural network model for Street Fighter II
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(256, activation='relu'), # max(0, x)
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer - sigmoid for multi-label classification
        Dense(output_dim, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the neural network model
    """
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[CustomCallback()],
        verbose=1
    )
    
    return model, history

def train_and_save_xgboost(X_train, y_train, feature_columns, action_columns, model_dir='model'):
    """
    Train XGBoost models for multi-label classification and save them
    """
    try:
        import xgboost as xgb
        from sklearn.multioutput import MultiOutputClassifier
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print("\nTraining XGBoost models...")
        
        # Create base XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=150,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        
        # Wrap in MultiOutputClassifier for multi-label classification
        multi_target_model = MultiOutputClassifier(xgb_model)
        multi_target_model.fit(X_train, y_train)
        
        # Save the complete model
        model_path = os.path.join(model_dir, 'xgboost_model.pkl')
        joblib.dump(multi_target_model, model_path)
        print(f"Saved XGBoost model to {model_path}")
        
        return multi_target_model
        
    except ImportError:
        print("\nXGBoost not installed. Please install with: pip install xgboost")
        return None
    except Exception as e:
        print(f"\nError training XGBoost: {str(e)}")
        return None

def evaluate_xgboost(model, X_test, y_test, action_columns):
    """
    Evaluate XGBoost model on multi-label classification
    """
    if model is None:
        print("No XGBoost model provided for evaluation")
        return
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("\nEvaluating XGBoost model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Metrics per label
        print("\nPer-action Metrics:")
        results = []
        for i, action in enumerate(action_columns):
            precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
            
            print(f"{action:15} Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
            results.append({
                'action': action,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return results
        
    except Exception as e:
        print(f"\nError evaluating XGBoost: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, action_columns):
    """
    Evaluate the trained model
    """
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate accuracy for each action
    action_accuracy = np.mean(y_pred_binary == y_test, axis=0)
    
    # Print action-specific accuracy
    for i, action in enumerate(action_columns):
        print(f"{action} accuracy: {action_accuracy[i]:.4f}")
    
    # Calculate F1 score for each action
    from sklearn.metrics import f1_score
    f1_scores = []
    for i in range(y_test.shape[1]):
        f1 = f1_score(y_test[:, i], y_pred_binary[:, i])
        f1_scores.append(f1)
        print(f"{action_columns[i]} F1 score: {f1:.4f}")
    
    return action_accuracy, f1_scores

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot the training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history plot saved to {save_path}")

def analyze_button_combinations(y_train):
    """
    Analyze the most common button combinations in the training data
    """
    # Convert to strings for easier counting
    button_combinations = [''.join(map(str, row)) for row in y_train]
    
    # Count occurrences
    from collections import Counter
    combo_counts = Counter(button_combinations)
    
    # Get the most common combinations
    most_common = combo_counts.most_common(10)
    
    print("\nMost common button combinations:")
    for combo, count in most_common:
        percentage = (count / len(button_combinations)) * 100
        print(f"Combination: {combo}, Count: {count}, Percentage: {percentage:.2f}%")
    
    return combo_counts

def save_trained_model(model, scaler, feature_columns, action_columns, model_dir='model'):
    """
    Save the trained model and related artifacts
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model
    model_path = os.path.join(model_dir, 'sf2_model.h5')
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature and action columns
    columns_path = os.path.join(model_dir, 'columns.npz')
    np.savez(columns_path, 
             feature_columns=feature_columns,
             action_columns=action_columns)
    print(f"Column information saved to {columns_path}")

def main():
    # Configuration
    data_file = 'game_data.csv'  # Path to dataset
    epochs = 100
    batch_size = 64
    test_size = 0.2
    model_dir = 'model'
    
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler, feature_columns, action_columns = load_and_preprocess_data(
            data_file, test_size
        )
        
        # Analyze button combinations
        analyze_button_combinations(y_train)

        xgb_model = train_and_save_xgboost(X_train, y_train, feature_columns, action_columns, model_dir)
        if xgb_model:
            evaluate_xgboost(xgb_model, X_test, y_test, action_columns)
        
        # Build the model
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        model = build_model(input_dim, output_dim)
        
        # Print model summary
        model.summary()
        
        # Train the model
        model, history = train_model(
            model, X_train, y_train, X_test, y_test, 
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test, action_columns)
        
        # Plot training history
        plot_training_history(history)
        
        # Save the model and related artifacts
        save_trained_model(model, scaler, feature_columns, action_columns, model_dir)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        
        # Try to diagnose the issue with the dataset
        try:
            print("\nAttempting to diagnose dataset issues...")
            with open(data_file, 'r') as f:
                first_few_lines = [f.readline() for _ in range(5)]
            
            print("First few lines of the dataset:")
            for i, line in enumerate(first_few_lines):
                print(f"Line {i+1}: {line[:100]}... (truncated)")
                
            print("\nDetecting delimiter...")
            delimiters = [',', '\t', ';', '|']
            for delimiter in delimiters:
                counts = [line.count(delimiter) for line in first_few_lines]
                print(f"Delimiter '{delimiter}': counts per line = {counts}")
                
            print("\nSuggestion: Make sure your CSV file is properly formatted with consistent delimiters.")
            print("If using tab-delimited data, ensure there are no extra tabs or formatting issues.")
        except Exception as diag_error:
            print(f"Error during diagnosis: {str(diag_error)}")

if __name__ == "__main__":
    main()

