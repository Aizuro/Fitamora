import cv2
import numpy as np
import mediapipe as mp
import joblib
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define which landmarks to use
EXCLUDED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_THUMB, mp_pose.PoseLandmark.RIGHT_THUMB,
]

USEFUL_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT, mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
]

def extract_body_features_only(image_path):
    """
    Extract pose landmarks features using MediaPipe, excluding hand landmarks
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None, None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    features = []
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract only useful landmarks (excluding hands)
        for landmark_enum in USEFUL_LANDMARKS:
            if landmark_enum not in EXCLUDED_LANDMARKS:
                landmark = landmarks[landmark_enum.value]
                features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate additional body posture features
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        # Additional features
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        hip_diff = abs(left_hip.y - right_hip.y)
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        shoulder_hip_ratio = shoulder_center_y / hip_center_y if hip_center_y != 0 else 1
        head_shoulder_ratio = nose.y / shoulder_center_y if shoulder_center_y != 0 else 1
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        torso_width_ratio = shoulder_width / hip_width if hip_width != 0 else 1
        knee_diff = abs(left_knee.y - right_knee.y)
        
        features.extend([
            shoulder_diff, hip_diff, shoulder_hip_ratio, 
            head_shoulder_ratio, torso_width_ratio, knee_diff
        ])
        
        return np.array(features), results.pose_landmarks, image_rgb
    else:
        print(f"No pose detected in: {image_path}")
        return None, None, None

def load_model(model_path):
    """Load trained model"""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

def predict_single_image(model, image_path, label_names, confidence_threshold=0.5):
    """
    TRUE Multi-label prediction for a single image
    """
    features, landmarks, image = extract_body_features_only(image_path)
    
    if features is None:
        print("Failed to extract features from the image.")
        return None, None
    
    # Check feature dimension
    if hasattr(model.estimators_[0], 'n_features_in_'):
        expected_features = model.estimators_[0].n_features_in_
        if features.shape[0] != expected_features:
            print(f"Feature dimension mismatch! Expected {expected_features}, got {features.shape[0]}")
            # Adjust features to match
            if features.shape[0] > expected_features:
                features = features[:expected_features]
            else:
                features = np.pad(features, (0, expected_features - features.shape[0]), 'constant')
    
    features = features.reshape(1, -1)
    
    # TRUE MULTI-LABEL PREDICTION
    predictions = []
    probabilities = []
    
    for i, estimator in enumerate(model.estimators_):
        proba = estimator.predict_proba(features)[0]
        pred = 1 if proba[1] >= confidence_threshold else 0
        predictions.append(pred)
        probabilities.append(proba[1])  # Probability of class 1
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"MULTI-LABEL PREDICTION RESULTS: {image_path}")
    print(f"{'='*60}")
    
    results = {}
    detected_problems = []
    
    for i, label_name in enumerate(label_names):
        prob = probabilities[i]
        pred = predictions[i]
        
        results[label_name] = {
            'prediction': bool(pred),
            'confidence': prob,
            'status': 'DETECTED' if pred == 1 else 'NORMAL'
        }
        
        if pred == 1:
            detected_problems.append(f"{label_name} ({prob:.3f})")
        
        status_color = '\033[91m' if pred == 1 else '\033[92m'
        reset_color = '\033[0m'
        
        print(f"{label_name:15}: {status_color}{'DETECTED' if pred == 1 else 'NORMAL':8}{reset_color} "
              f"(confidence: {prob:.3f})")
    
    # Multi-label summary
    print(f"\n{'─'*40}")
    if detected_problems:
        print("MULTI-LABEL DIAGNOSIS:")
        for problem in detected_problems:
            print(f"  ⚠️  {problem}")
    else:
        print("✅ NORMAL POSTURE - No problems detected")
    print(f"{'─'*40}")
    
    # Visualize results
    visualize_multi_label_prediction(image, landmarks, results, label_names, image_path)
    
    return results, features

def visualize_multi_label_prediction(image, landmarks, results, label_names, image_path):
    """
    Visualize multi-label prediction results
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw landmarks
    if landmarks:
        for landmark_enum in USEFUL_LANDMARKS:
            if landmark_enum not in EXCLUDED_LANDMARKS:
                landmark = landmarks.landmark[landmark_enum.value]
                h, w, _ = image_bgr.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image_bgr, (cx, cy), 4, (0, 255, 0), -1)
    
    # Prepare result text
    detected_labels = [label for label in label_names if results[label]['prediction']]
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_position = 30
    
    cv2.putText(image_bgr, "POSTURE ANALYSIS RESULTS", (10, y_position), font, 0.7, (255, 255, 255), 2)
    y_position += 30
    
    if detected_labels:
        cv2.putText(image_bgr, "DETECTED PROBLEMS:", (10, y_position), font, 0.6, (0, 0, 255), 2)
        y_position += 25
        
        for i, label in enumerate(detected_labels):
            conf = results[label]['confidence']
            text = f"{i+1}. {label} ({conf:.2f})"
            cv2.putText(image_bgr, text, (20, y_position + i*25), font, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(image_bgr, "NORMAL POSTURE", (10, y_position), font, 0.6, (0, 255, 0), 2)
        y_position += 25
    
    # Add confidence scores for all labels
    y_position += 30
    cv2.putText(image_bgr, "ALL CONFIDENCE SCORES:", (10, y_position), font, 0.5, (255, 255, 0), 1)
    y_position += 20
    
    for i, label in enumerate(label_names):
        conf = results[label]['confidence']
        color = (0, 0, 255) if results[label]['prediction'] else (0, 255, 0)
        text = f"{label}: {conf:.3f}"
        cv2.putText(image_bgr, text, (20, y_position + i*20), font, 0.4, color, 1)
    
    # Save result
    output_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
    cv2.imwrite(output_path, image_bgr)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    title = "Multi-Label Posture Analysis\n"
    if detected_labels:
        title += "Detected: " + ", ".join(detected_labels)
    else:
        title += "Normal Posture"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"Result image saved as: {output_path}")

def batch_predict_multi_label(model, image_folder, label_names, confidence_threshold=0.5):
    """
    Multi-label prediction for all images in a folder
    """
    import glob
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for extension in image_extensions:
        image_paths.extend(glob.glob(f"{image_folder}/{extension}"))
    
    print(f"Found {len(image_paths)} images for multi-label prediction")
    
    all_results = []
    
    for image_path in image_paths:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {image_path}")
        print(f"{'='*80}")
        
        results, features = predict_single_image(model, image_path, label_names, confidence_threshold)
        
        if results:
            all_results.append({
                'image_path': image_path,
                'results': results,
                'features': features
            })
    
    # Summary report
    print(f"\n{'#'*80}")
    print("BATCH PREDICTION SUMMARY")
    print(f"{'#'*80}")
    
    for result in all_results:
        detected = [label for label in label_names if result['results'][label]['prediction']]
        print(f"{result['image_path']}: {len(detected)} problems - {detected if detected else 'Normal'}")
    
    return all_results

# Main execution
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "posture_classifier_no_hands.pkl"
    LABEL_NAMES = ["fh", "kypho", "normal", "pt"]  # Your actual labels
    
    # Load model
    model = load_model(MODEL_PATH)
    
    if model is None:
        print("Please train the model first or check the model path.")
    else:
        print("Multi-label model loaded successfully!")
        print("Available labels:", LABEL_NAMES)
        
        # Prediction mode
        print("\nChoose prediction mode:")
        print("1. Predict single image")
        print("2. Predict all images in a folder")
        print("3. Adjust confidence threshold and predict single image")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if image_path:
                predict_single_image(model, image_path, LABEL_NAMES)
        
        elif choice == "2":
            folder_path = input("Enter folder path: ").strip()
            if folder_path:
                batch_predict_multi_label(model, folder_path, LABEL_NAMES)
        
        elif choice == "3":
            image_path = input("Enter image path: ").strip()
            try:
                threshold = float(input("Enter confidence threshold (0.0-1.0): ").strip())
                if 0 <= threshold <= 1:
                    predict_single_image(model, image_path, LABEL_NAMES, threshold)
                else:
                    print("Threshold must be between 0 and 1")
            except ValueError:
                print("Invalid threshold value")
        
        else:
            print("Invalid choice")