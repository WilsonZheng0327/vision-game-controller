import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ======= CONFIGURATION (MODIFY THESE VARIABLES) =======
# Path to the trained model (.pth file)
MODEL_PATH = "model/best_model.pth"

# Path to a single image or directory of images
IMAGE_PATH = "data/HaGRIDv2_dataset_512/dislike/0000f1d1-1b41-4489-bdb4-1e43ec991c81.jpg"  # Can be a single image or a directory
IMAGE_PATH = "me_tests"

# Number of classes in your model
NUM_CLASSES = 12

# Show top K predictions
TOP_K = 3

# Directory to save visualization results
OUTPUT_DIR = "inference_results"
# =====================================================

def load_model(model_path, num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def get_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def preprocess_image(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor, img

def predict(model, img_tensor, class_names, top_k=3):
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top k probabilities and class indices
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Convert to lists
        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        # Get class names for top predictions
        if isinstance(top_indices, int):  # Handle case when top_k=1
            top_indices = [top_indices]
            top_probs = [top_probs]
            
        top_classes = [class_names[idx] for idx in top_indices]
    
    return list(zip(top_classes, top_probs))

def visualize_prediction(img, predictions, image_name, output_dir):
    # Create figure with the image and predictions
    plt.figure(figsize=(10, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display the predictions
    plt.subplot(1, 2, 2)
    
    # Create bar chart
    classes = [p[0] for p in predictions]
    probs = [p[1] for p in predictions]
    y_pos = np.arange(len(classes))
    
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Top Predictions')
    
    # Save and show the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_prediction.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def process_single_image(model, image_path, transform, class_names, top_k, output_dir):
    img_tensor, img = preprocess_image(image_path, transform)
    predictions = predict(model, img_tensor, class_names, top_k)
    
    # Print predictions
    print(f"\nPredictions for {os.path.basename(image_path)}:")
    for i, (class_name, prob) in enumerate(predictions):
        print(f"{i+1}. {class_name}: {prob:.4f}")
    
    # Visualize predictions
    visualize_prediction(img, predictions, os.path.basename(image_path), output_dir)

def process_directory(model, dir_path, transform, class_names, top_k, output_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            process_single_image(model, file_path, transform, class_names, top_k, output_dir)

def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, NUM_CLASSES)
    
    # Load class names
    class_names = ["dislike", "fist", "four", "like", "no_gesture", "ok", "one", "palm", "rock", "three", "three_gun", "three2"]
    
    # Get image transform
    transform = get_transform()
    
    # Check if image path is a file or directory
    if os.path.isfile(IMAGE_PATH):
        # Process single image
        process_single_image(model, IMAGE_PATH, transform, class_names, TOP_K, OUTPUT_DIR)
    elif os.path.isdir(IMAGE_PATH):
        # Process all images in directory
        print(f"Processing all images in {IMAGE_PATH}...")
        process_directory(model, IMAGE_PATH, transform, class_names, TOP_K, OUTPUT_DIR)
    else:
        print(f"Error: {IMAGE_PATH} is not a valid file or directory.")

if __name__ == "__main__":
    main()