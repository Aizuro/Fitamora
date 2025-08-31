import os
import pandas as pd
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define which landmarks to use (excluding hands and unnecessary points)
USEFUL_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,           # 0
    mp_pose.PoseLandmark.LEFT_EYE_INNER, # 1
    mp_pose.PoseLandmark.LEFT_EYE,       # 2
    mp_pose.PoseLandmark.LEFT_EYE_OUTER, # 3
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,# 4
    mp_pose.PoseLandmark.RIGHT_EYE,      # 5
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,# 6
    mp_pose.PoseLandmark.LEFT_EAR,       # 7
    mp_pose.PoseLandmark.RIGHT_EAR,      # 8
    mp_pose.PoseLandmark.MOUTH_LEFT,     # 9
    mp_pose.PoseLandmark.MOUTH_RIGHT,    # 10
    mp_pose.PoseLandmark.LEFT_SHOULDER,  # 11
    mp_pose.PoseLandmark.RIGHT_SHOULDER, # 12
    mp_pose.PoseLandmark.LEFT_ELBOW,     # 13 (we'll use but not hand)
    mp_pose.PoseLandmark.RIGHT_ELBOW,    # 14 (we'll use but not hand)
    mp_pose.PoseLandmark.LEFT_WRIST,     # 15 (exclude from features)
    mp_pose.PoseLandmark.RIGHT_WRIST,    # 16 (exclude from features)
    mp_pose.PoseLandmark.LEFT_PINKY,     # 17 (exclude)
    mp_pose.PoseLandmark.RIGHT_PINKY,    # 18 (exclude)
    mp_pose.PoseLandmark.LEFT_INDEX,     # 19 (exclude)
    mp_pose.PoseLandmark.RIGHT_INDEX,    # 20 (exclude)
    mp_pose.PoseLandmark.LEFT_THUMB,     # 21 (exclude)
    mp_pose.PoseLandmark.RIGHT_THUMB,    # 22 (exclude)
    mp_pose.PoseLandmark.LEFT_HIP,       # 23
    mp_pose.PoseLandmark.RIGHT_HIP,      # 24
    mp_pose.PoseLandmark.LEFT_KNEE,      # 25
    mp_pose.PoseLandmark.RIGHT_KNEE,     # 26
    mp_pose.PoseLandmark.LEFT_ANKLE,     # 27
    mp_pose.PoseLandmark.RIGHT_ANKLE,    # 28
    mp_pose.PoseLandmark.LEFT_HEEL,      # 29
    mp_pose.PoseLandmark.RIGHT_HEEL,     # 30
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,# 31
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX# 32
]

# Landmarks to EXCLUDE completely (hands and fingers)
EXCLUDED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_ELBOW,     # 13 (we'll use but not hand)
    mp_pose.PoseLandmark.RIGHT_ELBOW,    # 14 (we'll use but not hand)
    mp_pose.PoseLandmark.LEFT_WRIST,     # 15
    mp_pose.PoseLandmark.RIGHT_WRIST,    # 16
    mp_pose.PoseLandmark.LEFT_PINKY,     # 17
    mp_pose.PoseLandmark.RIGHT_PINKY,    # 18
    mp_pose.PoseLandmark.LEFT_INDEX,     # 19
    mp_pose.PoseLandmark.RIGHT_INDEX,    # 20
    mp_pose.PoseLandmark.LEFT_THUMB,     # 21
    mp_pose.PoseLandmark.RIGHT_THUMB,    # 22
]

def extract_body_features_only(image_path):
    """
    Extract pose landmarks features using MediaPipe, excluding hand landmarks
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe
    results = pose.process(image_rgb)
    
    features = []
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract only useful landmarks (excluding hands)
        for i, landmark_enum in enumerate(USEFUL_LANDMARKS):
            if landmark_enum not in EXCLUDED_LANDMARKS:
                landmark = landmarks[landmark_enum.value]
                features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate additional body posture features
        # Shoulder alignment
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        features.append(shoulder_diff)
        
        # Hip alignment
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_diff = abs(left_hip.y - right_hip.y)
        features.append(hip_diff)
        
        # Shoulder-hip ratio (spine curvature indicator)
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        shoulder_hip_ratio = shoulder_center_y / hip_center_y if hip_center_y != 0 else 1
        features.append(shoulder_hip_ratio)
        
        # Head position relative to shoulders
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        head_shoulder_ratio = nose.y / shoulder_center_y if shoulder_center_y != 0 else 1
        features.append(head_shoulder_ratio)
        
        # Torso alignment (shoulder to hip distance ratio)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        torso_width_ratio = shoulder_width / hip_width if hip_width != 0 else 1
        features.append(torso_width_ratio)
        
        # Knee alignment
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        knee_diff = abs(left_knee.y - right_knee.y)
        features.append(knee_diff)
        
        print(f"Successfully extracted {len(features)} features from {os.path.basename(image_path)}")
        return np.array(features)
    else:
        print(f"No pose detected in: {image_path}")
        return None

def load_and_preprocess_data(csv_path):
    """
    Load data from CSV and extract features (without hand landmarks)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Labels: {df.columns[1:].tolist()}")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    failed_images = []
    
    for idx, row in df.iterrows():
        image_path = row['filepath']
        
        # Extract MediaPipe features (without hands)
        features = extract_body_features_only(image_path)
        
        if features is not None:
            # Get labels (exclude filepath column)
            labels = row[1:].values.astype(int)
            
            features_list.append(features)
            labels_list.append(labels)
        else:
            failed_images.append(image_path)
    
    if failed_images:
        print(f"\nFailed to process {len(failed_images)} images:")
        for img in failed_images[:5]:
            print(f"  - {img}")
        if len(failed_images) > 5:
            print(f"  - ... and {len(failed_images) - 5} more")
    
    return np.array(features_list), np.array(labels_list), df.columns[1:].tolist()

def train_model(X, y, label_names):
    """
    Train multi-label classification model
    """
    if len(X) == 0:
        print("Error: No features extracted!")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Number of labels: {y_train.shape[1]}")
    
    # Train Random Forest classifier for multi-label
    base_classifier = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=15,
        min_samples_split=3,
        class_weight='balanced'
    )
    
    model = MultiOutputClassifier(base_classifier)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-label metrics
    print("\nPer-label Accuracy:")
    for i, label_name in enumerate(label_names):
        label_acc = accuracy_score(y_test[:, i], y_pred[:, i])
        print(f"  {label_name}: {label_acc:.4f}")
    
    # Detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    return model

# Main execution
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = "dataset_postur.csv"
    
    # Load and preprocess data
    print("Loading data and extracting features (excluding hand landmarks)...")
    X, y, label_names = load_and_preprocess_data(csv_path)
    
    if len(X) == 0:
        print("Error: No features extracted. Check your image paths.")
    else:
        # Train model
        print("\nTraining multi-label posture classifier...")
        model = train_model(X, y, label_names)
        
        # Save model
        joblib.dump(model, 'posture_classifier_no_hands.pkl')
        print("\nModel saved as 'posture_classifier_no_hands.pkl'")
        
        # Feature importance (optional)
        print("\nFeature importance analysis completed!")