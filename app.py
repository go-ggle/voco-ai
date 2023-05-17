from scipy.io.wavfile import write, read

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

from storages import upload_file

app = Flask(__name__)
api = Api(app)

@app.route("/tts", methods=["POST"])
def tts():
    #numpy array representing the audio data
    print("tts")
    params = request.get_json()
    language = params['language']
    text = params['text']
    user_id = int(params['voiceId'])
    team_id = request.args.get("teamId")
    project_id = request.args.get("projectId")
    block_id = request.args.get("blockId")

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
    req = dict(request.form)
    audio = request.files['file']
    print(type(audio))

    #STT
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio) as source:
        audio_text = recognizer.listen(source)
    stt_text = recognizer.recognize_google(audio_text)
    print(stt_text)
    text_id = request.args.get("textId")
    train_text = linecache.getline("train.txt", int(text_id))
    nlp = spacy.load("en_core_web_md")
    doc1 = nlp(stt_text.lower())
    doc2 = nlp(train_text.lower())
    print(similarity)

    if similarity < 0.8:
        return jsonify("유사도 통과 몬함"), 205

    user_id = request.args.get("userId")
    save_path = './vc/Data/p' + str(user_id)

    if not os.path.isdir(save_path):
        #data dir 생성
        os.makedirs(save_path, exist_ok=True)
        shutil.copy('./vc/Data/train_list.txt', save_path)
        shutil.copy('./vc/Data/val_list.txt', save_path)

    audio.save(save_path + "/" + text_id + ".wav")
    #s = io.BytesIO(audio.encode())

    #data, samplerate = sf.read(io.BytesIO(au))
    #sf.write(save_path + '/' + str(text_id) + '.wav', data, 24000)

    #with open(save_path + '/' + str(text_id) + '.wav', mode='wb') as f:
    #    f.write(audio)

    #input = request.files['audio']
    #input.save(save_path + '/' + str(text_id) + '.wav')

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
    app.run(host='192.168.0.6', port=5000, debug=True)
