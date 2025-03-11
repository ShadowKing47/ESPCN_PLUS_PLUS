"""
ESPCN Super Resolution - Complete Usage Example

This script demonstrates the complete workflow for training and using the ESPCN model:
1. Train the model on a dataset of high-resolution images
2. Run inference on test images
3. Measure performance metrics

Usage:
1. Set the directories in the script below
2. Run the script to train the model and perform inference
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

# Import the modules we created
from espcn_model import build_and_compile_model
from espcn_dataset import create_dataset, train_model
from espcn_inference import batch_inference

# Set directories and parameters
HR_TRAIN_DIR = r"C:/Users/91995/Downloads/div2k/DIV2K_valid_HR/DIV2K_valid_HR"  # Directory with high-resolution training images
TEST_DIR = r"C:\Users\91995\OneDrive\Desktop\ESCPN++\Set14\Set14"             # Directory with test images
OUTPUT_DIR = r"C:\Users\91995\OneDrive\Desktop\ESCPN++\Set14\Set14_E"                 # Directory to save model and results
HR_TEST_DIR = "path/to/hr_test_images"       # Optional directory with high-res versions of test images
SCALE_FACTOR = 4                             # Upscaling factor
BATCH_SIZE = 16                              # Batch size for training
PATCH_SIZE = 10                              # Size of training patches
EPOCHS = 50                                  # Number of training epochs

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
INFERENCE_DIR = os.path.join(OUTPUT_DIR, "inference_results")
os.makedirs(INFERENCE_DIR, exist_ok=True)

# Step 1: Create datasets for training
print("Creating training datasets...")
train_dataset, val_dataset = create_dataset(
    HR_TRAIN_DIR, 
    scale_factor=SCALE_FACTOR,
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE
)

# Step 2: Build and compile the model
print("Building model...")
model = build_and_compile_model(scale_factor=SCALE_FACTOR)
model.summary()

# Step 3: Train the model
print("Training model...")
history = train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=EPOCHS,
    output_dir=MODEL_DIR
)

# Step 4: Run inference on test images
print("Running inference on test images...")
model_path = os.path.join(MODEL_DIR, "espcn_final_model.h5")
batch_inference(
    TEST_DIR,
    INFERENCE_DIR,
    model_path,
    scale_factor=SCALE_FACTOR,
    original_hr_dir=HR_TEST_DIR
)

print("Complete! The results are saved in:", OUTPUT_DIR)
print("- Model and training history:", MODEL_DIR)
print("- Super-resolution results:", INFERENCE_DIR)

# Example of how to use the trained model for a single image
def quick_super_resolve(image_path, model_path, output_path):
    """Quick function to super-resolve a single image"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.convert('RGB')
    lr_img = np.array(img, dtype=np.float32) / 255.0
    
    # Predict
    lr_img_batch = np.expand_dims(lr_img, axis=0)
    sr_img_batch = model.predict(lr_img_batch, verbose=0)
    sr_img = sr_img_batch[0]
    
    # Save result
    sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
    sr_img_pil.save(output_path)
    
    # Display comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Input ({img.width}x{img.height})")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Super-Resolution ({sr_img_pil.width}x{sr_img_pil.height})")
    plt.imshow(sr_img_pil)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return sr_img_pil

# Example usage of the quick function:
# quick_super_resolve(
#     "path/to/single/image.jpg",
#     os.path.join(MODEL_DIR, "espcn_final_model.h5"),
#     "super_resolved_output.png"
# )