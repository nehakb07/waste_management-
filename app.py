import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Create uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map classes to their names
classes = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.route('/')
def index():
    return render_template('index.html', predicted_class=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    img_array /= 255.0

    # Set the input tensor
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, img_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)

    # Convert output to class predictions
    predictions = np.argmax(output_data, axis=1)
    predicted_class = classes[predictions[0]]

    # Remove the previous image from the uploaded folder
    if os.path.exists(img_path):
        os.remove(img_path)

    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
