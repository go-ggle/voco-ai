from flask import Flask, jsonify, request, send_file
from flask_restx import Api
from scipy.io.wavfile import write, read
from torch import nn

import sys
import io
import os
import shutil
import soundfile as sf
import json
import linecache
from pydub import AudioSegment

from vc import *
from vc import Utils
from vc.metadata import CreateMetadata
from vc.train import Train
from vc.vc_inference import Inference

import speech_recognition as sr
from gtts import gTTS

import librosa
import spacy
import wave
from pydub import AudioSegment
import numpy as np

from storages import *

app = Flask(__name__)
api = Api(app)

@app.route("/tts", methods=["POST"])
def tts():
    #numpy array representing the audio data
    print("tts")
    languages = ['en', 'en', 'en', 'es', 'fr', 'it', 'de', 'ru', 'ar', 'zh', 'ja', 'id'] 
    domains = ['us', 'co.uk', 'co.in', 'es', 'fr']

    params = request.get_json()

    user_id = int(params['voiceId'])
    team_id = request.args.get("teamId")
    project_id = request.args.get("projectId")
    block_id = request.args.get("blockId")

    language_id = int(params['language'])
    language = languages[language_id]
    text = params['text']

    if(language_id<5):
        domain = domains[language_id]
        tts = gTTS(text=text, lang=language, tld=domain)
    else:
        tts = gTTS(text=text, lang=language)

    filename = str(block_id) + ".wav"
    tts.save(str(filename))

    audio, source_sr  = librosa.load(block_id + ".wav", sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32
    wave_audio = Inference(audio, user_id).inference()
    os.remove(str(block_id + '.wav'))

    upload_file(wave_audio, user_id, team_id, project_id, block_id)

    #return wav file
    #return send_file(byte_io, mimetype="audio/x-wav")
    return send_file('vc/output_audio_dir/converted_p400.wav')

@app.route("/put_data", methods=["POST"])
def put_data():
    req = dict(request.files)
    audio = req['file']
    user_id = request.args.get("userId")
    text_id = request.args.get("textId")
    
    dir_path = './vc/Data/p' + str(user_id)
    pcm_path = dir_path + '/' + str(text_id) + '.pcm'
    wav_path = dir_path + '/' + str(text_id) + '.wav'

    audio.save(pcm_path)
    cmd = "ffmpeg -y -f s16le -ar 24k -ac 2 -i " + pcm_path + " " + wav_path
    os.system(cmd)
    os.remove(pcm_path)
   
    if not os.path.isdir(dir_path):
        #data dir 생성
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy('./vc/Data/train_list.txt', dir_path)
        shutil.copy('./vc/Data/val_list.txt', dir_path)
    
    #STT
    recognizer = sr.Recognizer()

    with sr.AudioFile(wav_path) as source:
        audio_text = recognizer.listen(source)
    stt_text = recognizer.recognize_google(audio_text, show_all=True)
    print(stt_text)
    train_text = linecache.getline("train.txt", int(text_id))
    
    if type(stt_text) is list:
        return jsonify("유사도 통과 실패"), 205

    nlp = spacy.load("en_core_web_md")
    doc1 = nlp(stt_text['alternative'][0]['transcript'].lower())
    doc2 = nlp(train_text.lower())
    similarity = doc1.similarity(doc2)
    print(similarity)

    if similarity < 0.1:
        os.remove(wav_path)
        return jsonify("유사도 통과 몬함"), 205

    response = {"textId": text_id}
    return jsonify(response), 200


@app.route("/audios", methods=["GET"])
def get_audio():
    user_id = request.args.get("user_id")
    text_id = request.args.get("text_id")
    file_path = 'vc/Data/p' + str(user_id) + '/' + str(text_id) + '.wav'
    print(file_path)
    if(os.path.exists(file_path) == False):
        return Response("파일 없음", status=404)

    return send_file('vc/Data/p' + str(user_id) + '/' + str(text_id) + '.wav')


@app.route("/train", methods=["POST"])
def train():
    user_id = request.args.get("user_id")

    #model dir 생성
    model_dst = './vc/Models/p' + str(user_id)
    os.makedirs(model_dst, exist_ok=True)

    #config dir 생성
    config_dst = './vc/Configs/p' + str(user_id)
    os.makedirs(config_dst, exist_ok=True)
    shutil.copy('./vc/Configs/config.yml', config_dst)

    CreateMetadata(user_id).write_metadata()

    Train(user_id).train()

    response = {"userId": user_id, "isRegistered": True}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
