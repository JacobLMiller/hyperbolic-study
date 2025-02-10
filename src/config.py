import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MONGOPASS = os.environ.get('MONGOPASS')
    LIVE_VERSION = os.environ.get('LIVE_VERSION')
    SESSIONPASS = os.environ.get('SESSIONPASS')