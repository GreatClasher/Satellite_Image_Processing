import os
import cv2

from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect, url_for, g, current_app

import time

from predict import predict


app = Flask(__name__)
app.debug = True


@app.route("/")
def hello():
  return render_template('./home.html')

@app.route("/about/")
def about():
  return render_template('./about.html')

@app.route("/main/")
def main():
  return render_template('./main.html')

@app.route('/upload_file/', methods = ['POST'])
def upload_file():
    UPLOAD_FOLDER = './static/inputimages'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        else:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('main.html', user_image = filename)


@app.route('/prediction/<file>', methods=['GET', 'POST'])
def prediction(file):
  models_path = ['resnet34.h5']
  # start = time.time()
  image = cv2.imread("./static/inputimages/"+file,0)
  
  with current_app.app_context():
    ans = predict(image,models_path)
    result_image_path = './models_masks.jpeg'
  return redirect(url_for('result', result_image=result_image_path))


@app.route('/result/<result_image>', methods=['GET', 'POST'])
def result(result_image):
    return render_template('result.html', result_image_path=result_image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
