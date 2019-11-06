import os


class Config():
    consumer_key = os.environ.get("CONSUMER_KEY") or 'XXX'
    consumer_secrete = os.environ.get("CONSUMER_SECRETE") or 'XXX'
    access_key = os.environ.get("ACCESS_KEY") or 'XXX'
    access_secrete = os.environ.get("ACCESS_SECRETE") or 'XXX'