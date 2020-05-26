import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow_core.examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
import numpy as np
from Segementation_utils import pre_process_data, display
from Segmentation_models import model_HairMatteNet, model_simple_segmentation
import os.path

# Inputs
input_height = 224
input_width = 224
learning_rate = 0.01
batch_size = 5
training_epochs = 30

my_path = os.path.abspath(os.path.dirname(__file__))
training_path = os.path.join(my_path, "Training folder")


# Images generator
processed_images, processed_labels = pre_process_data(training_path, input_height, input_width) #"C://Workspaces//ARCproject//Image_postpro//Training folder"

X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(processed_images, processed_labels, test_size=0.2)

X_train = np.concatenate([arr[np.newaxis] for arr in X_train_list])/255.0
X_test = np.concatenate([arr[np.newaxis] for arr in X_test_list])/255.0
y_train = np.concatenate([arr[np.newaxis] for arr in y_train_list])
y_test = np.concatenate([arr[np.newaxis] for arr in y_test_list])


# Model definition

# # Simple model
# img_input = Input(shape=[input_height, input_width, 3])
# model_output = model_simple_segmentation(img_input)
# model = keras.Model(inputs=img_input, outputs=model_output, name="simple_CNN")

# # HairMatteNet
img_input = Input(shape=[input_height, input_width, 3])
model_output = model_HairMatteNet(img_input)
model = keras.Model(inputs=img_input, outputs=model_output, name="HairMatteNet")

# # Unet
# img_input = Input(shape=[input_width, input_height, 3])
# base_model = tf.keras.applications.MobileNetV2(input_shape=[input_width, input_height, 3], include_top=False)
# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',   # 64x64
#     'block_3_expand_relu',   # 32x32
#     'block_6_expand_relu',   # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',      # 4x4
# ]
#
# layers = [base_model.get_layer(name).output for name in layer_names]
# down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
# down_stack.trainable = True
# up_stack = [
#     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
#     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
#     pix2pix.upsample(128, 3),  # 16x16 -> 32x32
#     pix2pix.upsample(64, 3),   # 32x32 -> 64x64
# ]
#
#
# model = unet_model(img_input)

model.summary()


# Train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  # ['accuracy']

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(X_test, y_test))

# Save model
# model.save('HairMatteNet.h5') # Saved in Keras

# Convert the model to tensorflow lite.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open( 'tflite_HairMatteNet.tflite' , 'wb' ).write( tflite_model )

# Display results
for i in range( X_test.shape[0]):
    y_pred = model.predict(X_test[i][np.newaxis])
    y_pred = y_pred.reshape((input_width, input_height, 2))
    image_list = []
    image_list.append(X_test[i])
    image_list.append(y_test[i])
    image_list.append(np.argmax(y_pred, axis=-1))
    display(image_list)


print("bye!")


