import tempfile
import sys
import os
import errno
import json
import re
import imdb

from nltk.corpus import stopwords
from flask import Flask, request, abort, make_response, jsonify
from flask_restful import reqparse, abort, Api, Resource

import requests

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

IMG_SIZE = 224

POSTER_PREDICT_LABELS = [
    'Action',
    'Adventure',
    'Animation',
    'Biography',
    'Comedy',
    'Crime',
    'Drama',
    'Family',
    'Fantasy',
    'History',
    'Horror',
    'Music',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
]


DESCRIPTION_PREDICT_LABELS = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                              'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                              'Short', 'Sport', 'Thriller', 'War', 'Western']


app = Flask(__name__)
api = Api(app)

## Load 2 Models Prepare for Predict

# Computer Vision Model
# poster_model = tf.keras.models.load_model("model_20200829.h5", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

place_model = tf.keras.models.load_model("model_ResNetV2.h5")

# NLP Model
# description_model = pickle.load(open('model_description_20200831.pkl', 'rb'))
# tf1 = pickle.load(open("tfidf1.pkl", 'rb'))


@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn (default)".format(
        version
    )
    return message

# CV Predict Pipeline (Function)
def poster_predict(image_path, isUrl=False):
    if isUrl:
        img_path = tf.keras.utils.get_file(fname=next(
            tempfile._get_candidate_names()), origin=image_path)
    else:
        img_path = image_path

    img = keras.preprocessing.image.load_img(
        img_path, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array/255
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Generate prediction
    predict_value = poster_model.predict(img_array)
    prediction = (predict_value > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction = prediction[prediction == 1].index.values

    os.remove(img_path)

    response = {}
    response['predict_genres'] = [
        f'{POSTER_PREDICT_LABELS[p]}: {predict_value.tolist()[0][p]:.2f}' for p in prediction]

    return response


def place_predict(image_path, isUrl=False):
    if isUrl:
        img_path = tf.keras.utils.get_file(fname=next(
            tempfile._get_candidate_names()), origin=image_path)
    else:
        img_path = image_path

    img = keras.preprocessing.image.load_img(
        img_path, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array/255
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Generate prediction
    predict = place_model.predict(img_array)
    result = np.argmax(predict, axis=1)

    dict_mapping = {
        0: 'คาเฟ่ / ร้านกาแฟ / ร้านขนม / เบเกอรี่',
        1: 'ผับ บาร์ สถานที่ท่องเที่ยวยามค่ำคืน',
        2: 'ร้านอาหาร',
        3: 'เน้นกิจกรรม เช่น สวนสนุก สวนน้ำ ฟาร์ม พิพิธภัณฑ์',
        4: 'เน้นช๊อปปิ้ง เช่น ตลาด / ตลาดกลางคืน / สตรีทฟูดส์ / ห้างสรรพสินค้า',
        5: 'เน้นทะเล / ชายหาด / กิจกรรมทางน้ำ',
        6: 'เน้นประวัติศาสตร์ / วัฒนธรรม',
        7: 'เน้นป่า / ภูเขา / จุดชมวิว',
        8: 'เน้นสักการะสิ่งศักดิ์สิทธิ์ เช่น วัด / ศาลเจ้า / โบสถ์'
    }

    category_list = np.vectorize(dict_mapping.get)(result).tolist()

    os.remove(img_path)

    response = {}
    response['category'] = category_list

    return response

# NLP Predict Pipeline (Function)
def description_predict(description):
    # NLP Model
    # description_model = pickle.load(open('model_description_20200831.pkl', 'rb'))
    tf1 = pickle.load(open("tfidf1.pkl", 'rb'))

    tfidf_vectorizer = TfidfVectorizer(vocabulary=tf1.vocabulary_)

    # clean text using regex
    description = re.sub("[^a-zA-Z]", " ", description)
    # remove whitespaces
    description = ' '.join(description.split())
    # convert text to lowercase
    description = description.lower()

    no_stopword_text = [w for w in description.split() if w not in stop_words]
    description = ' '.join(no_stopword_text)

    description_vec = tfidf_vectorizer.fit_transform([description])

    predict_value = description_model.predict(description_vec)
    prediction = (predict_value > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction = prediction[prediction == 1].index.values

    response = {}
    response['predict_genres'] = [
        f'{DESCRIPTION_PREDICT_LABELS[p]}: {predict_value.tolist()[0][p]:.2f}' for p in prediction]

    return response


# ### Create RESTful APIs Structure using Flask-RESTful ###

class Imdb(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('titleId', type=str, required=True,
                    help="Please Specify Title Id !!!")
        super(Imdb, self).__init__()

    def get(self):
        args = self.reqparse.parse_args()
        title_id = args['titleId']
        print(f"got title_id {title_id}")

        # creating instance of IMDb
        ia = imdb.IMDb()
        movie = ia.get_movie(int(title_id[2:]))

        response = {}
        response['id'] = title_id
        response['title'] = movie['title']
        response['actual_imdb_genres'] = movie['genres']
        response['description'] = movie['plot'][0].split('::')[0]
        movie_img_url = movie['full-size cover url']

        response['poster_predict_genres'] = dict(poster_predict(
            movie_img_url, isUrl=True))['predict_genres']
        response['description_predict_genres'] = dict(description_predict(
            str(response['description'])))['predict_genres']

        return response


class ImageGenre(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('imgurl', required=True,
                    help="Please Specify Image Url !!!")
        super(ImageGenre, self).__init__()

    def get(self):
        args = self.reqparse.parse_args()
        image_url = args['imgurl']
        response = {}
        response['source'] = image_url
        response['predict_genres'] = dict(poster_predict(
            image_url, isUrl=True))['predict_genres']
        return response


class TextGenre(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, help="Please Specify Text !!!")
        super(TextGenre, self).__init__()

    def get(self):
        args = self.reqparse.parse_args()
        text = args['text']
        response = {}
        response['source'] = text
        response['predict_genres'] = dict(description_predict(text))[
            'predict_genres']
        return response

class ImagePlace(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('imgurl', required=True,
                    help="Please Specify Image Url !!!")
        super(ImageGenre, self).__init__()

    def get(self):
        args = self.reqparse.parse_args()
        image_url = args['imgurl']
        response = {}
        response['source'] = image_url
        response['predict_genres'] = dict(poster_predict(
            image_url, isUrl=True))['predict_genres']
        return response


##
# Actually setup the Api resource routing here
##
api.add_resource(Imdb, '/imdb')
api.add_resource(ImageGenre, '/genre/image')
api.add_resource(TextGenre, '/genre/text')
api.add_resource(ImagePlace, '/place/image')
