import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Then import TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the InceptionV3 model pre-trained on ImageNet data
base_cnn_model = InceptionV3(weights='imagenet')
feature_extractor_model = keras.Model(inputs=base_cnn_model.input, outputs=base_cnn_model.layers[-2].output)

# Load and preprocess the image
def load_and_process_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

# Generate image features using the InceptionV3 model
def generate_image_features(image_path):
    processed_image = load_and_process_image(image_path)
    image_features = feature_extractor_model.predict(processed_image)
    return image_features

# Load and preprocess the captions
captions_list = ["a person standing in front of a building", "a cat sitting on a windowsill", "a group of people walking on the beach"]
caption_tokenizer = Tokenizer()
caption_tokenizer.fit_on_texts(captions_list)
vocabulary_size = len(caption_tokenizer.word_index) + 1

# Convert captions to sequences of integers
caption_sequences = caption_tokenizer.texts_to_sequences(captions_list)

# Pad sequences to have a consistent length
max_caption_length = max(len(seq) for seq in caption_sequences)
padded_caption_sequences = pad_sequences(caption_sequences, maxlen=max_caption_length, padding='post')

# Define the model for caption generation
embedding_dimension = 256
rnn_units = 512

# Image feature extraction model
image_features_input = layers.Input(shape=(2048,))
image_embedding_layer = layers.Dense(embedding_dimension)(image_features_input)
image_embedding_layer = layers.RepeatVector(max_caption_length)(image_embedding_layer)

# Caption sequence model
caption_input_layer = layers.Input(shape=(max_caption_length,))
caption_embedding_layer = layers.Embedding(vocabulary_size, embedding_dimension)(caption_input_layer)
caption_lstm_layer = layers.LSTM(rnn_units, return_sequences=True)(caption_embedding_layer)
caption_output_layer = layers.TimeDistributed(layers.Dense(embedding_dimension))(caption_lstm_layer)

# Combine image and caption embeddings
merged_layers = layers.Concatenate(axis=-1)([image_embedding_layer, caption_output_layer])

# Use a dense layer to generate the final caption
final_output_layer = layers.TimeDistributed(layers.Dense(vocabulary_size, activation='softmax'))(merged_layers)

# Define and compile the model
caption_generator_model = keras.Model(inputs=[image_features_input, caption_input_layer], outputs=final_output_layer)
caption_generator_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with your data (you'll need image data and corresponding caption data)
# caption_generator_model.fit([image_data, caption_data], caption_labels, epochs=epochs, batch_size=batch_size)
