import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from .MainImplementation import predictor
from .Services import MainServices

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/linuxsagar/tempTest'
app.config['GRAPH_FOLDER'] = '/home/linuxsagar/PycharmProjects/FruitClassifierTesting/graphs'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    MainServices.clean_dir()
    MainServices.clean_graph_dir()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    MainServices.clean_graph_dir()
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('index.html')


@app.route('/predict')
def predict():
    data = predictor()
    MainServices.clean_dir()
    return render_template('index.html', result=data[0], hue=data[1], edge=data[2], haar=data[3])


@app.errorhandler(Exception)
def all_exception_handler(error):
    return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)