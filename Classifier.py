#%%
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_data(pairs, labels, images, crop_size=64):
    data_images = []
    data_coords = []
    for idx, ((x1, y1), (x2, y2)) in enumerate(pairs):
        image = images[idx]
        
        # Calculate the center point between the two beads
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Define the bounding box coordinates
        half_size = crop_size / 2
        min_x = int(mid_x - half_size)
        min_y = int(mid_y - half_size)
        max_x = int(mid_x + half_size)
        max_y = int(mid_y + half_size)

        # Ensure coordinates are within image boundaries
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image.width, max_x)
        max_y = min(image.height, max_y)

        # Crop and resize the image
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        resized_image = cropped_image.resize((crop_size, crop_size))
        image_array = np.array(resized_image)

        # Convert to grayscale if needed
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        elif image_array.shape[2] == 3:
            # Convert to grayscale
            image_array = np.mean(image_array, axis=2, keepdims=True)

        # Normalize pixel values
        image_array = image_array.astype('float32') / 255.0
        data_images.append(image_array)

        # Normalize bead coordinates within the cropped image
        coord_x1 = (x1 - min_x) / (max_x - min_x)
        coord_y1 = (y1 - min_y) / (max_y - min_y)
        coord_x2 = (x2 - min_x) / (max_x - min_x)
        coord_y2 = (y2 - min_y) / (max_y - min_y)

        data_coords.append([coord_x1, coord_y1, coord_x2, coord_y2])

    data_images = np.array(data_images)
    data_coords = np.array(data_coords, dtype='float32')
    labels = np.array(labels, dtype='float32')
    return data_images, data_coords, labels

def create_model(input_shape):
    # Image input branch
    image_input = Input(shape=input_shape, name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)

    # Coordinate input branch
    coord_input = Input(shape=(4,), name='coord_input')
    y = layers.Dense(32, activation='relu')(coord_input)
    y = layers.Dense(32, activation='relu')(y)

    # Combine image features and coordinates
    combined = layers.concatenate([x, y])

    # Fully connected layers
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dense(1, activation='sigmoid')(z)  # Output layer

    model = models.Model(inputs=[image_input, coord_input], outputs=z)
    return model


#%% 
input_shape = (64, 64, 1)  # Adjust channels if necessary
model = create_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have lists of pairs, labels, and images
# pairs = [((x1, y1), (x2, y2)), ...]
# labels = [1, 0, 1, ...]
# images = [Image.open(image_path) for image_path in image_paths]

# Prepare the data
X_images, X_coords, y = prepare_data(pairs, labels, images, crop_size=64)

#%% Split into training and validation sets
X_img_train, X_img_val, X_coord_train, X_coord_val, y_train, y_val = train_test_split(
    X_images, X_coords, y, test_size=0.2, random_state=42)


#%% Train the model
history = model.fit(
    {'image_input': X_img_train, 'coord_input': X_coord_train},
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(
        {'image_input': X_img_val, 'coord_input': X_coord_val},
        y_val
    )
)

#%% Plot accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(
    {'image_input': X_img_val, 'coord_input': X_coord_val},
    y_val
)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
#%% 
def predict_connection(pair, image):
    X_image, X_coord, _ = prepare_data([pair], [0], [image], crop_size=64)  # Label is a placeholder
    prediction = model.predict({'image_input': X_image, 'coord_input': X_coord})
    predicted_label = int(prediction[0] > 0.5)
    return predicted_label  # 1 for connected, 0 for not connected