import tensorflow as tf
import numpy as np
import os
import time
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model(model_path):
    """Load the trained ESPCN model"""
    return tf.keras.models.load_model(model_path)

def process_image(image_path, model, scale_factor=3):
    """
    Process a single image with the ESPCN model
    
    Args:
        image_path: Path to the input image
        model: Trained ESPCN model
        scale_factor: Upscaling factor
        
    Returns:
        sr_image: Super-resolution image as numpy array [0, 1]
        elapsed_time: Processing time in seconds
    """
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Preprocess
    lr_img = np.array(img, dtype=np.float32) / 255.0
    
    # Measure inference time
    start_time = time.time()
    
    # Add batch dimension
    lr_img_batch = np.expand_dims(lr_img, axis=0)
    
    # Predict
    sr_img_batch = model.predict(lr_img_batch, verbose=0)
    
    # Remove batch dimension
    sr_img = sr_img_batch[0]
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    return sr_img, elapsed_time

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return tf.image.psnr(
        tf.convert_to_tensor(img1, dtype=tf.float32),
        tf.convert_to_tensor(img2, dtype=tf.float32),
        max_val=1.0
    ).numpy()

def batch_inference(input_dir, output_dir, model_path, scale_factor=3, original_hr_dir=None):
    """
    Perform inference on all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        model_path: Path to trained model
        scale_factor: Upscaling factor
        original_hr_dir: Optional directory with original high-res images for PSNR calculation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    # Get list of images
    input_dir = pathlib.Path(input_dir)
    image_paths = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.jpeg'))
    
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Process each image
    times = []
    psnr_values = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Process image
        sr_img, elapsed_time = process_image(img_path, model, scale_factor)
        times.append(elapsed_time)
        
        # Save output image
        output_filename = os.path.join(output_dir, f"SR_{img_path.name}")
        sr_img_pil = Image.fromarray((sr_img * 255).astype(np.uint8))
        sr_img_pil.save(output_filename)
        
        # Calculate PSNR if original HR images are available
        if original_hr_dir:
            hr_path = os.path.join(original_hr_dir, img_path.name)
            if os.path.exists(hr_path):
                hr_img = Image.open(hr_path)
                hr_img = hr_img.resize((sr_img.shape[1], sr_img.shape[0]), Image.BICUBIC)
                hr_img = np.array(hr_img, dtype=np.float32) / 255.0
                psnr = calculate_psnr(sr_img, hr_img)
                psnr_values.append(psnr)
    
    # Statistics
    avg_time = sum(times) / len(times)
    total_time = sum(times)
    
    print("\n===== Results Summary =====")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing time per image: {avg_time:.2f} seconds")
    
    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Min PSNR: {min(psnr_values):.2f} dB")
        print(f"Max PSNR: {max(psnr_values):.2f} dB")
    
    # Plot statistics
    plt.figure(figsize=(12, 8))
    
    # Processing time plot
    plt.subplot(2, 1, 1)
    plt.bar(range(len(times)), times)
    plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Average: {avg_time:.3f}s')
    plt.title('Processing Time per Image')
    plt.xlabel('Image Index')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    # PSNR plot if available
    if psnr_values:
        plt.subplot(2, 1, 2)
        plt.bar(range(len(psnr_values)), psnr_values)
        plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Average: {avg_psnr:.2f} dB')
        plt.title('PSNR per Image')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR (dB)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_stats.png'))
    
    # Save a sample comparison
    if len(image_paths) > 0:
        sample_path = image_paths[0]
        
        # Original low-resolution image
        lr_img = Image.open(sample_path)
        
        # Super-resolution result
        sr_path = os.path.join(output_dir, f"SR_{sample_path.name}")
        sr_img = Image.open(sr_path)
        
        # Display comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title(f"Input ({lr_img.width}x{lr_img.height})")
        plt.imshow(lr_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Output ({sr_img.width}x{sr_img.height})")
        plt.imshow(sr_img)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_comparison.png'))
    
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ESPCN Inference on a directory of images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--scale_factor', type=int, default=3, help='Upscaling factor')
    parser.add_argument('--hr_dir', type=str, default=None, help='Directory with original HR images for PSNR calculation')
    
    args = parser.parse_args()
    
    batch_inference(
        args.input_dir,
        args.output_dir,
        args.model_path,
        args.scale_factor,
        args.hr_dir
    )