import os
from flask import Flask, flash, request, redirect, jsonify
from werkzeug.utils import secure_filename
import torch
import time
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, ViTForImageClassification
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
UPLOAD_FOLDER = './temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(12).hex()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

feature_extractor = ViTFeatureExtractor.from_pretrained("nateraw/food")
model = ViTForImageClassification.from_pretrained("nateraw/food")
model.eval()
labels = model.config.id2label

def predict(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    m = torch.nn.Softmax(dim=1)

    outputs = m(outputs.logits)[0].tolist()

    prediction = [{"score": outputs[x], "label": labels[x]} for x in range(0, len(outputs))]
    # preds = pd.DataFrame({"food":labels, "score":outputs})
    prediction = sorted(prediction, key=lambda x: x['score'], reverse=True)
    return prediction[:3]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def standard():
    return "please upload an image to /predict"

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            print(request)
            return 'No file part'
        file = request.files['file']
        # if user does not selec
        # t file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print(request)
            return 'No selected file'
        if file and allowed_file(file.filename):
            app.logger.info('file transfered: %s ', file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            startTime = time.time()
            preds = predict(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
            executionTime = (time.time() - startTime)
            app.logger.info('prediction took: %s ',  str(executionTime))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(preds)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


