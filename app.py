import base64
import numpy as np
import io
from PIL import Image  # import keras
# from  tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from flask import request
from flask import jsonify
from flask import Flask




app = Flask(__name__)

def get_model():
    global model
    model = load_model('CNN7.h5')
    print(" * Model loaded!")
		
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)	
    image=preprocess_input(image)
    return image

print(" * Loading Keras model...")
get_model()

    

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force = True)    
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224,224))  
    prediction = model.predict(processed_image).tolist()	
    response = {
        'prediction':{
            'aa' : prediction[0][0],
            'Advantage' : prediction[0][1],
            'PointCut' : prediction[0][2],
            'Four' : prediction[0][3],
            'Penalty' : prediction[0][4],
            'Three' : prediction[0][5],
			'Two' : prediction[0][6]
            }
        }
    return jsonify(response)

if __name__=="__main__":
    app.run()
