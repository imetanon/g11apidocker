import tempfile
import sys
import os
import errno
import json
import re
import imdb
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    MemberJoinedEvent, MemberLeftEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, SpacerComponent, IconComponent, ButtonComponent,
    SeparatorComponent, QuickReply, QuickReplyButton,
    ImageSendMessage)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot import (
    LineBotApi, WebhookHandler
)
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, abort
from flask_restful import reqparse, abort, Api, Resource

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

## Load 2 Models Prepare for Predict

# # Computer Vision Model
# poster_model = tf.keras.models.load_model(
#     "model_20200829.h5", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})

# # NLP Model
# description_model = pickle.load(open('model_description_20200831.pkl', 'rb'))
# tf1 = pickle.load(open("tfidf1.pkl", 'rb'))


global poster_model
# save default graph in a global var
global graph
graph = tf.Graph()
with graph.as_default():
    poster_model = tf.keras.models.load_model("model_20200829.h5", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})


app = Flask(__name__)
api = Api(app)

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# function for create tmp dir for download content


def make_static_tmp_dir():
    try:
        os.makedirs(static_tmp_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
            pass
        else:
            raise


# create tmp dir for download content
make_static_tmp_dir()


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    if text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' +
                                    profile.display_name),
                    TextSendMessage(text='Status message: ' +
                                    str(profile.status_message))
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))

    else:
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=event.message.text))

# # Other Message Type


# @handler.add(MessageEvent, message=(ImageMessage))
# def handle_content_message(event):
#     if isinstance(event.message, ImageMessage):
#         ext = 'jpg'
#     else:
#         return

#     message_content = line_bot_api.get_message_content(event.message.id)
#     with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
#         for chunk in message_content.iter_content():
#             tf.write(chunk)
#         tempfile_path = tf.name

#     dist_path = tempfile_path + '.' + ext
#     dist_name = os.path.basename(dist_path)
#     os.rename(tempfile_path, dist_path)

#     predict_message = json.dumps(poster_predict(
#         os.path.join('static', 'tmp', dist_name)))

#     line_bot_api.reply_message(
#         event.reply_token, [
#             # TextSendMessage(text='Save content.'),
#             # TextSendMessage(text=request.host_url + \
#             #                 os.path.join('static', 'tmp', dist_name)),
#             TextSendMessage(text=predict_message)
#         ])


@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn (default)".format(
        version
    )
    return message

# CV Predict Pipeline (Function)
def poster_predict(image_path, isUrl=False):
    # poster_model = tf.keras.models.load_model("model_20200829.h5", compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    if isUrl:
        img_path = tf.keras.utils.get_file(fname=next(
            tempfile._get_candidate_names()), origin=image_path)

    img = keras.preprocessing.image.load_img(
        img_path, color_mode='rgb', target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array/255
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Generate prediction
    with graph.as_default():
        predict_value = poster_model.predict(img_array)
    prediction = (predict_value > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction = prediction[prediction == 1].index.values

    os.remove(img_path)

    response = {}
    response['predict_genres'] = [
        f'{POSTER_PREDICT_LABELS[p]}: {predict_value.tolist()[0][p]:.2f}' for p in prediction]

    return response

# NLP Predict Pipeline (Function)
def description_predict(description):
    # NLP Model
    description_model = pickle.load(open('model_description_20200831.pkl', 'rb'))
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
    def get(self, title_id):
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


##
# Actually setup the Api resource routing here
##
# api.add_resource(Imdb, '/imdb/<title_id>')
api.add_resource(ImageGenre, '/genre/image')
api.add_resource(TextGenre, '/genre/text')
