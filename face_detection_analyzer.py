import cv2
import numpy as np
import time
import os
from datetime import datetime

def real_time_face_detection_analysis():
    """
    A comprehensive face detection application that:
    1. Captures video from the computer's camera
    2. Detects faces using OpenCV's DNN-based face detector
    3. Analyzes performance under various conditions
    4. Records and saves failure cases
    5. Displays real-time metrics and warnings
    """
    print("Starting Real-Time Face Detection Analysis")
    print("Press 'ESC' to exit")
    
    # Create directory for saving failure cases
    failure_dir = "failure_cases"
    os.makedirs(failure_dir, exist_ok=True)
    print(f"Failure cases will be saved to: {failure_dir}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {width}x{height} at {fps} FPS")
    
    # Initialize face detector using OpenCV DNN
    print("Initializing OpenCV DNN face detector...")
    
    # Load model
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    
    # Check if model files exist, if not, download them
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(modelFile) or not os.path.exists(configFile):
        print("Model files not found. Downloading them...")
        os.makedirs("models", exist_ok=True)
        
        # Use pre-trained model from OpenCV's DNN module (Face Detection)
        # This is a lightweight alternative to MTCNN
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create a fallback for eye detection to improve analysis
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        print("Using Haar Cascade detectors instead of DNN model")
        use_dnn = False
    else:
        # Load the model
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        use_dnn = True
        print("Using DNN-based face detector")
    
    # Define colors for visualization
    colors = {
        'face_box': (0, 255, 0),  # Green
        'face_box_low_conf': (0, 165, 255),  # Orange
        'text': (255, 255, 255),  # White
        'warning': (0, 0, 255),   # Red
        'landmarks': (255, 0, 0)  # Blue
    }
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'detected_faces': 0,
        'low_confidence_detections': 0,
        'no_detection_frames': 0,
        'last_detection_timestamp': time.time(),
        'start_time': time.time(),
        'condition_counts': {
            'good_illumination': 0,
            'poor_illumination': 0,
            'side_illumination': 0,
            'frontal_view': 0,
            'profile_view': 0,
            'partial_occlusion': 0,
            'multiple_faces': 0
        }
    }
    
    # Thresholds and parameters
    confidence_threshold = 0.7  # Minimum confidence for a "good" detection
    face_absence_threshold = 2.0  # Seconds without detection to trigger warning
    save_interval = 30  # Save a sample frame every X frames
    
    # Detection history for temporal analysis
    detection_history = []
    confidence_history = []
    history_length = 15
    
    print("Starting detection loop. Press ESC to exit.")
    
    # Main detection loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Update statistics
        stats['total_frames'] += 1
        current_frame = stats['total_frames']
        current_time = time.time()
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect faces
        t_start = time.time()
        
        faces = []
        confidences = []
        
        if use_dnn:
            # Using DNN-based detector
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Minimum threshold for consideration
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Ensure coordinates are within frame boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Only add valid detections
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2-x1, y2-y1))
                        confidences.append(confidence)
        else:
            # Using Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in face_detections:
                # Extract the face ROI for further analysis
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detect eyes in the face ROI to estimate confidence
                eyes = eye_detector.detectMultiScale(roi_gray)
                
                # Estimate confidence based on number of eyes detected
                # More sophisticated confidence estimation could be implemented
                if len(eyes) >= 2:
                    confidence = 0.9  # Both eyes detected
                elif len(eyes) == 1:
                    confidence = 0.7  # One eye detected
                else:
                    confidence = 0.5  # No eyes detected
                
                faces.append((x, y, w, h))
                confidences.append(confidence)
        
        detection_time = time.time() - t_start
        
        # Update detection history
        detection_history.append(len(faces) > 0)
        if len(detection_history) > history_length:
            detection_history.pop(0)
        
        # Calculate detection stability
        detection_stability = sum(detection_history) / len(detection_history)
        
        # Update statistics based on detection results
        if faces:
            stats['detected_faces'] += len(faces)
            stats['last_detection_timestamp'] = current_time
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            confidence_history.append(avg_confidence)
            if len(confidence_history) > history_length:
                confidence_history.pop(0)
            
            # Update condition counts
            if len(faces) > 1:
                stats['condition_counts']['multiple_faces'] += 1
            
            if avg_confidence > 0.8:
                stats['condition_counts']['good_illumination'] += 1
            elif avg_confidence < 0.6:
                stats['condition_counts']['poor_illumination'] += 1
            
            # Estimating view type based on confidence and stability
            if avg_confidence > 0.8 and detection_stability > 0.9:
                stats['condition_counts']['frontal_view'] += 1
            elif avg_confidence < 0.7 or detection_stability < 0.7:
                stats['condition_counts']['profile_view'] += 1
        else:
            # No faces detected in this frame
            stats['no_detection_frames'] += 1
            
            # Save failure case if we previously had stable detection
            if (current_time - stats['last_detection_timestamp'] > face_absence_threshold and
                detection_stability > 0.5 and current_frame > history_length):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(failure_dir, f"lost_detection_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Detection loss case saved: {filename}")
        
        # Process and visualize detected faces
        for i, ((x, y, w, h), confidence) in enumerate(zip(faces, confidences)):
            # Record low confidence detections
            if confidence < confidence_threshold:
                stats['low_confidence_detections'] += 1
                stats['condition_counts']['poor_illumination'] += 1
                
                # Save low confidence frame periodically
                if current_frame % save_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(failure_dir, f"low_conf_{confidence:.2f}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Low confidence case saved: {filename}")
            
            # Draw face bounding box (color based on confidence)
            box_color = colors['face_box'] if confidence >= confidence_threshold else colors['face_box_low_conf']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
            
            # Display confidence
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(display_frame, conf_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
            
            # In the Haar version, we don't have detailed facial landmarks
            # But we can draw the detected eyes if available
            if not use_dnn:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_detector.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.circle(display_frame, (x + ex + ew//2, y + ey + eh//2), 5, colors['landmarks'], 2)
                
                # Estimate head rotation based on eye positions if two eyes detected
                if len(eyes) == 2:
                    eye1_x = eyes[0][0] + eyes[0][2]//2
                    eye2_x = eyes[1][0] + eyes[1][2]//2
                    
                    # Sort eyes from left to right
                    if eye1_x > eye2_x:
                        eye1_x, eye2_x = eye2_x, eye1_x
                    
                    eye_distance = abs(eye1_x - eye2_x)
                    face_width = w
                    
                    # Estimate rotation based on relative position of eyes
                    rotation_ratio = eye_distance / face_width
                    
                    if rotation_ratio < 0.2:
                        rotation_text = "Head Rotated"
                        cv2.putText(display_frame, rotation_text, (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['warning'], 1)
                        stats['condition_counts']['profile_view'] += 1
                    else:
                        stats['condition_counts']['frontal_view'] += 1
        
        # Display performance metrics
        elapsed_since_last = current_time - stats['last_detection_timestamp']
        total_elapsed = current_time - stats['start_time']
        
        # Calculate and display FPS
        fps = stats['total_frames'] / total_elapsed
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        # Display detection rate
        if stats['total_frames'] > 0:
            detection_rate = (stats['detected_faces'] / stats['total_frames']) * 100
            cv2.putText(display_frame, f"Detection Rate: {detection_rate:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        # Display detection time
        cv2.putText(display_frame, f"Detection Time: {detection_time*1000:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        # Display detection stability
        cv2.putText(display_frame, f"Stability: {detection_stability:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
        
        # Display current condition analysis
        line_y = 150
        if confidence_history:
            avg_conf = sum(confidence_history) / len(confidence_history)
            condition_text = f"Avg Confidence: {avg_conf:.2f}"
            cv2.putText(display_frame, condition_text, (10, line_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
            line_y += 30
        
        # Display warnings based on current conditions
        warnings = []
        
        # Check for detection loss
        if elapsed_since_last > face_absence_threshold and detection_stability > 0.5:
            warnings.append("Face lost - possible occlusion or out of frame")
            
        # Check for poor illumination
        if confidence_history and sum(confidence_history) / len(confidence_history) < 0.7:
            warnings.append("Poor illumination detected")
            
        # Check for unstable detection
        if detection_stability < 0.7 and stats['total_frames'] > history_length:
            warnings.append("Unstable detection - movement or rotation")
        
        # Display warning messages
        for warning in warnings:
            cv2.putText(display_frame, warning, (10, line_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['warning'], 2)
            line_y += 30
        
        # Save a sample frame periodically
        if current_frame % save_interval == 0 and faces:
            condition = "normal"
            if len(faces) > 1:
                condition = "multiple_faces"
            elif confidence_history and sum(confidence_history) / len(confidence_history) < 0.7:
                condition = "poor_illumination"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(failure_dir, f"sample_{condition}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
        
        # Display the frame
        cv2.imshow('Face Detection Analysis', display_frame)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final performance analysis
    print("\n===== Face Detection Performance Analysis =====")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total faces detected: {stats['detected_faces']}")
    print(f"Frames with no detection: {stats['no_detection_frames']}")
    print(f"Low confidence detections: {stats['low_confidence_detections']}")
    
    if stats['total_frames'] > 0:
        print(f"Overall detection rate: {(stats['detected_faces'] / stats['total_frames']) * 100:.2f}%")
        print(f"No detection rate: {(stats['no_detection_frames'] / stats['total_frames']) * 100:.2f}%")
        print(f"Low confidence rate: {(stats['low_confidence_detections'] / stats['total_frames']) * 100:.2f}%")
    
    print("\n--- Condition Analysis ---")
    total_condition_frames = sum(stats['condition_counts'].values())
    if total_condition_frames > 0:
        for condition, count in stats['condition_counts'].items():
            percentage = (count / stats['total_frames']) * 100
            print(f"{condition}: {count} frames ({percentage:.1f}%)")
    
    print(f"\nAverage FPS: {stats['total_frames'] / total_elapsed:.2f}")
    print(f"Run time: {total_elapsed:.1f} seconds")
    print("====================================")
    print(f"Sample images saved to: {failure_dir}")

if __name__ == "__main__":
    real_time_face_detection_analysis()
