import librosa
import sqlite3
import numpy as np
from flask import Flask, render_template, request, session
import joblib
import os
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)
model = joblib.load('models/knn_model.pkl')

# Database setup
DATABASE_FILENAME = 'audio_classifier.sqlite'


def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS audio_classification (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           file_name TEXT,
                           audio_file BLOB,
                           predicted_label TEXT,
                           correct_label TEXT
                       )''')
    conn.commit()
    conn.close()


create_table()


def extract_feature(file_name=None, feature=None):
    if file_name:
        print('Extracting', file_name)
        X, sample_rate = librosa.load(file_name)
        if X.ndim > 1:
            X = X[:, 0]
        X = X.T

        if feature == 'mfccs':
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            return mfccs
        elif feature == 'chroma':
            chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=40).T, axis=0)
            return chroma
        elif feature == 'mel':
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=40).T, axis=0)
            return mel


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        audio_file = request.files['audio_file']
        audio_path = os.path.join('audio', audio_file.filename)
        audio_file.save(audio_path)

        feature = extract_feature(audio_path, feature='mfccs')
        feature = feature.reshape(1, -1)

        prediction = model.predict(feature)
        print(prediction)
        command = prediction[0]

        session['file_name'] = audio_file.filename
        session['predicted_label'] = prediction[0]
        return render_template('index.html', command=command)
    else:
        return render_template('index.html')


@app.route('/improve', methods=['POST'])
def improve():
    correct_label = request.form['correct_label']

    # Retrieve file name and predicted label from session
    file_name = session.get('file_name')
    predicted_label = session.get('predicted_label')

    # Read the temporarily stored audio file
    audio_path = os.path.join('audio', file_name)
    with open(audio_path, 'rb') as file:
        audio_data = file.read()

    # Store everything in the database
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO audio_classification (file_name, audio_file, predicted_label, correct_label) VALUES (?, ?, ?, ?)',
        (file_name, audio_data, predicted_label, correct_label))
    conn.commit()
    conn.close()

    # Optionally, delete the temporary audio file
    os.remove(audio_path)

    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
