import tensorflow as tf
import numpy as np
import os
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def create_dataset(hr_dir, scale_factor=3, patch_size=96, batch_size=16, validation_split=0.2):
    """
    Create a dataset from high-resolution images
    
    Args:
        hr_dir: Directory containing high-resolution images
        scale_factor: Upscaling factor (default: 3)
        patch_size: Size of extracted patches (default: 96)
        batch_size: Batch size (default: 16)
        validation_split: Fraction of images for validation (default: 0.2)
        
    Returns:
        train_dataset, val_dataset
    """
    hr_dir = pathlib.Path(hr_dir)
    hr_images = list(hr_dir.glob('*.png')) + list(hr_dir.glob('*.jpg')) + list(hr_dir.glob('*.jpeg'))
    
    # Shuffle and split for training and validation
    np.random.shuffle(hr_images)
    split_idx = int(len(hr_images) * (1 - validation_split))
    train_hr_images = hr_images[:split_idx]
    val_hr_images = hr_images[split_idx:]
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_hr_images)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_hr_images)
    
    # Map functions for processing
    def process_path(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    
    def random_crop(img):
        # Ensure image is large enough for cropping
        shape = tf.shape(img)
        min_dim = tf.minimum(shape[0], shape[1])
        if min_dim < patch_size:
            # Resize if image is too small
            scale = (patch_size + 10) / tf.cast(min_dim, tf.float32)
            new_height = tf.cast(tf.cast(shape[0], tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(shape[1], tf.float32) * scale, tf.int32)
            img = tf.image.resize(img, [new_height, new_width])
            shape = tf.shape(img)
        
        # Random crop
        hr_crop = tf.image.random_crop(img, [patch_size, patch_size, 3])
        return hr_crop
    
    def prepare_sample(hr):
        # Create low-resolution by downscaling high-resolution
        lr = tf.image.resize(hr, [patch_size // scale_factor, patch_size // scale_factor], 
                             method='bicubic')
        return lr, hr
    
    def augment(lr, hr):
        # Random flip
        if tf.random.uniform(()) > 0.5:
            lr = tf.image.flip_left_right(lr)
            hr = tf.image.flip_left_right(hr)
        
        # Random rotation
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        lr = tf.image.rot90(lr, k)
        hr = tf.image.rot90(hr, k)
        
        return lr, hr
    
    # Apply transformations
    train_dataset = (train_dataset
                     .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
                     .map(random_crop, num_parallel_calls=tf.data.AUTOTUNE)
                     .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
                     .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
                   .map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
                   .map(random_crop, num_parallel_calls=tf.data.AUTOTUNE)
                   .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, val_dataset

def display_samples(dataset, num_samples=3):
    """Display sample pairs of LR and HR images from the dataset"""
    plt.figure(figsize=(12, 4*num_samples))
    
    for i, (lr, hr) in enumerate(dataset.take(num_samples)):
        lr = lr[0].numpy()
        hr = hr[0].numpy()
        
        plt.subplot(num_samples, 2, i*2+1)
        plt.title(f"Low Resolution {lr.shape}")
        plt.imshow(lr)
        plt.axis('off')
        
        plt.subplot(num_samples, 2, i*2+2)
        plt.title(f"High Resolution {hr.shape}")
        plt.imshow(hr)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_dataset, val_dataset, epochs=50, output_dir='model_output'):
    """Train the ESPCN model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'espcn_model_{epoch:02d}.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=os.path.join(output_dir, 'logs'),
        update_freq='epoch'
    )
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint, lr_reducer, tensorboard]
    )
    
    # Save final model
    model.save(os.path.join(output_dir, 'espcn_final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr'], label='Training PSNR')
    plt.plot(history.history['val_psnr'], label='Validation PSNR')
    plt.title('PSNR during Training')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.show()
    
    return history

if __name__ == "__main__":
    from espcn_model import build_and_compile_model
    
    # Directory with high-resolution images
    hr_dir = r"C:/Users/91995/Downloads/div2k/DIV2K_valid_HR/DIV2K_valid_HR"
    
    # Create datasets
    train_dataset, val_dataset = create_dataset(hr_dir, scale_factor=3)
    
    # Display some samples
    display_samples(train_dataset)
    
    # Build and compile the model
    model = build_and_compile_model(scale_factor=3)
    
    # Train the model
    train_model(model, train_dataset, val_dataset, epochs=50, 
                output_dir='espcn_model_output')