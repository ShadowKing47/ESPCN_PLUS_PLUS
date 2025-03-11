import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Lambda, Add, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, LeakyReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply

def create_attention_module(inputs, filters):
    """Create a simple channel attention module"""
    attention = GlobalAveragePooling2D()(inputs)
    attention = Dense(filters // 4, activation='relu')(attention)
    attention = Dense(filters, activation='sigmoid')(attention)
    attention = Reshape((1, 1, filters))(attention)
    return Multiply()([inputs, attention])

def ESPCN(scale_factor=3, num_filters=64, attention=True):
    """
    Efficient Sub-Pixel Convolutional Neural Network model for super-resolution
    with added skip connections and attention mechanism
    
    Args:
        scale_factor: Upscaling factor (default: 3)
        num_filters: Number of filters in the convolution layers (default: 64)
        attention: Whether to use channel attention (default: True)
        
    Returns:
        Keras model
    """
    # Input layer
    inputs = Input(shape=(None, None, 3))
    
    # Convert to YUV color space if input is RGB
    x = Lambda(lambda x: tf.image.rgb_to_yuv(x))(inputs)
    
    # First convolutional layer (feature extraction)
    conv1 = Conv2D(num_filters, kernel_size=5, padding='same')(x)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    
    # Add attention if requested
    if attention:
        conv1 = create_attention_module(conv1, num_filters)
    
    # Second convolutional layer (feature extraction)
    conv2 = Conv2D(num_filters, kernel_size=3, padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    
    # Skip connection
    conv2_with_skip = Add()([conv2, conv1])
    
    # Third convolutional layer (feature extraction)
    conv3 = Conv2D(num_filters, kernel_size=3, padding='same')(conv2_with_skip)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    
    # Add attention if requested
    if attention:
        conv3 = create_attention_module(conv3, num_filters)
    
    # Skip connection
    conv3_with_skip = Add()([conv3, conv2_with_skip])
    
    # Final convolutional layer - map to 3 * scale_factor^2 feature maps
    # This is for sub-pixel convolution (pixel shuffle) for upscaling
    conv4 = Conv2D(3 * (scale_factor ** 2), kernel_size=3, padding='same')(conv3_with_skip)
    
    # Sub-pixel convolution (depth_to_space)
    # Rearranges data from depth to spatial dimensions for upscaling
    outputs = Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor))(conv4)
    
    # Convert back to RGB color space
    outputs = Lambda(lambda x: tf.image.yuv_to_rgb(x))(outputs)
    
    # Ensure output values are in range [0, 1]
    outputs = Lambda(lambda x: tf.clip_by_value(x, 0, 1))(outputs)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
def build_and_compile_model(scale_factor=3, learning_rate=0.001):
    model = ESPCN(scale_factor=scale_factor, num_filters=64, attention=True)
    
    # Compile model with Mean Squared Error loss and Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[tf.keras.metrics.MeanSquaredError(), 
                 tf.keras.metrics.RootMeanSquaredError(),
                 tf.keras.metrics.PSNR()]
    )
    
    return model

if __name__ == "__main__":
    # For testing - create and display the model summary
    model = build_and_compile_model(scale_factor=3)
    model.summary()