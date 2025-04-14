import torch
import cv2
import time
from PIL import Image
import json
import keyboard
import threading

from train import *

# ------------ HYPERPARAMETERS ------------
data_dir = 'data/HaGRIDv2_dataset_512'
model_type = 'efficientnet_b0'
model_path = "output/7747T9804V.pth"

use_edge_detection = False
edge_sigma = 1.0

key_mapping_path = "key_mappings.json"
camera_id = 0
interval = 0.1
confidence_threshold = 0.95
gesture_hold_frames = 3
output_file = "output/camera_output.txt"
debug = True

def load_key_mappings(path, class_names):
    with open(path, 'r') as f:
        mappings = json.load(f)
    
    # Check if all class names are in the mappings
    for class_name in class_names:
        if class_name not in mappings:
            print(f"Warning: Class '{class_name}' not found in mappings file. Adding with empty mapping.")
            mappings[class_name] = ""
    
    return mappings

class GestureController:
    def __init__(self, model, confidence_threshold, gesture_hold_frames, device, use_edge_detection, edge_sigma, transform, class_names, key_mappings, debug):
        self.confidence_threshold = confidence_threshold
        self.gesture_hold_frames = gesture_hold_frames
        self.model = model
        self.device = device
        self.use_edge_detection = use_edge_detection
        self.edge_sigma = edge_sigma
        self.transform = transform
        self.class_names = class_names
        self.key_mappings = key_mappings
        self.debug = debug
        
        # State tracking
        self.current_gesture = None
        self.current_gesture_count = 0
        self.last_triggered_gesture = None  # Track the last gesture that triggered a key
        self.active_keys = set()
        self.gesture_active = False  # Flag to track if a gesture is currently active
        self.last_prediction_time = 0
        self.running = True
        
        # Start keyboard controller thread
        self.controller_thread = threading.Thread(target=self.keyboard_controller_loop)
        self.controller_thread.daemon = True
        self.controller_thread.start()
    
    def process_frame(self, frame):
        """Process a frame and return prediction info"""
        # Apply edge detection if enabled
        if self.use_edge_detection:
            processed_frame = apply_edge_detection(frame, sigma=self.edge_sigma)
            cv2.imshow('Processed Frame (Edge Detection)', processed_frame)
        else:
            processed_frame = frame
        
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_frame_rgb)
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_batch)
        
        # Calculate probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the predicted class
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()
        
        top_prob = probabilities[predicted_class].item()
        
        return {
            'class_idx': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': top_prob,
            'frame': frame
        }
    
    def update_gesture_state(self, prediction):
        """Update the current gesture state based on prediction"""
        gesture_name = prediction['class_name']
        confidence = prediction['confidence']
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # No gesture detected - reset the active state
            if self.current_gesture is not None:
                # We were showing a gesture, but now we've stopped
                if self.debug:
                    print(f"Gesture ended: {self.current_gesture}")
                self.gesture_active = False
            
            # Reset counters
            self.current_gesture = None
            self.current_gesture_count = 0
            return None
        
        # Check if this is the same gesture as before
        if gesture_name == self.current_gesture:
            self.current_gesture_count += 1
        else:
            # New gesture detected
            self.current_gesture = gesture_name
            self.current_gesture_count = 1
            self.gesture_active = False  # Reset active flag for new gesture
        
        # Check if we've seen this gesture enough times consecutively
        if self.current_gesture_count >= self.gesture_hold_frames:
            # If the gesture is not yet marked as active, trigger it
            if not self.gesture_active:
                self.gesture_active = True
                self.last_triggered_gesture = self.current_gesture
                return self.current_gesture
            
            # Otherwise, gesture is already active, don't trigger again
            return None
        else:
            return None
    
    def trigger_keyboard_action(self, gesture):
        """Trigger keyboard action based on gesture"""
        if gesture is None or gesture not in self.key_mappings:
            return
        
        key = self.key_mappings[gesture]
        if not key:  # Skip empty mappings
            return
        
        # Set single active key (overwriting any previous key)
        self.active_keys = {key}
        
        if self.debug:
            print(f"Triggered: {gesture} → {key}")
    
    def release_all_keys(self):
        """Release all active keys"""
        self.active_keys.clear()
    
    def keyboard_controller_loop(self):
        """Thread that manages keyboard presses and releases - simplified version without modifiers"""
        while self.running:
            # Process active keys - one at a time, no modifiers
            keys_to_process = self.active_keys.copy()  # Make a copy to avoid race conditions
            for key in keys_to_process:
                try:
                    # Simple press and release without holding
                    keyboard.press_and_release(key)
                    if self.debug:
                        print(f"Pressed key: {key}")
                except Exception as e:
                    if self.debug:
                        print(f"Error pressing key {key}: {e}")
                self.active_keys.remove(key)  # Remove after processing
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def stop(self):
        """Stop the controller and release all keys"""
        self.running = False
        self.release_all_keys()
        if self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1.0)




def main():

    checkpoint = torch.load(model_path, map_location='cpu')

    # For EfficientNet models
    if model_type.startswith('efficientnet'):
        classifier_weight = [v for k, v in checkpoint.items() if 'classifier.1.weight' in k]
        if not classifier_weight:
            # Try alternate key patterns
            classifier_weight = [v for k, v in checkpoint.items() if 'classifier.weight' in k or 'fc.weight' in k]
        
        if classifier_weight:
            num_classes = classifier_weight[0].shape[0]
        else:
            # Fallback approach - use the last weight tensor's first dimension
            last_weight = list(checkpoint.values())[-2]  # -2 to avoid bias
            num_classes = last_weight.shape[0]

    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    print(f"Detected {num_classes} classes in the model")
    print([f"Class {i}" for i in class_names])
    
    key_mappings = load_key_mappings(key_mapping_path, class_names)
    print("Loaded gesture to key mappings:")
    for gesture, key in key_mappings.items():
        print(f"  {gesture} → {key or 'None'}")

    model = create_model(model_type, num_classes)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Get image transform
    _, val_transform = get_transforms()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return
    
    print("Camera opened successfully")
    print(f"Capturing an image every {interval} seconds")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Required consecutive frames: {gesture_hold_frames}")
    print("Press 'q' to quit")

    controller = GestureController(model, confidence_threshold, gesture_hold_frames, device, use_edge_detection, edge_sigma, val_transform, class_names, key_mappings, debug)

    last_capture_time = time.time() - interval

    try:
        with open(output_file, 'w') as file:
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture image from camera")
                    break
                
                # Display the frame
                # cv2.imshow('Camera Feed', frame)
                
                current_time = time.time()
                time_since_last_capture = current_time - last_capture_time
                
                if time_since_last_capture >= interval:
                    last_capture_time = current_time
                    
                    prediction = controller.process_frame(frame)
                    gesture = controller.update_gesture_state(prediction)
                    
                    if gesture:
                        controller.trigger_keyboard_action(gesture)


                    # Display results
                    prediction_text = f"Prediction: {prediction['class_name']} ({prediction['confidence']*100:.1f}%)"
                    status_text = f"Active: {'None' if not controller.current_gesture else controller.current_gesture}"
                    keys_text = f"Keys: {', '.join(controller.active_keys) if controller.active_keys else 'None'}"
                    
                    cv2.putText(
                        frame, 
                        prediction_text, 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                    
                    cv2.putText(
                        frame, 
                        status_text, 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 165, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                    
                    cv2.putText(
                        frame, 
                        keys_text, 
                        (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 165, 255), 
                        2, 
                        cv2.LINE_AA
                    )
                    
                    # Display the frame with prediction
                    cv2.imshow('Gesture Controller', frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
    finally:
        # Stop controller and release keys
        controller.stop()
        
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()
        
        print("Gesture controller stopped")

if __name__ == "__main__":
    main()