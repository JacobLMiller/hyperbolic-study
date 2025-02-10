from flask import Flask
from flask_bootstrap import Bootstrap5
from config import Config
from os.path import join, dirname, realpath, exists
from os import makedirs

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads/')

if not exists(UPLOAD_FOLDER): 
  makedirs(UPLOAD_FOLDER)
  

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap5(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from application import library, routes, errors
