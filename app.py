from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['STATIC_FOLDER'] = 'static'

model = load_model('classifier.h5')

def predict(image):
    image = image.resize((150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    prediction = model.predict(image)
    if prediction[0][0] < 0.5:
        return 'cat'
    else:
        return 'dog'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    image = Image.open(image_path)
    predicted_class = predict(image)
    return render_template('result.html', predicted_class=predicted_class, image_file=image_file)

if __name__ == '__main__':
    app.run()





