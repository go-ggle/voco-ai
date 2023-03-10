from flask import Flask, jsonify, request, send_file
from flask_restx import Api
from scipy.io.wavfile import write

import sys
import io
import os
import shutil
import soundfile as sf
import json
from pydub import AudioSegment

from utils import TextAudioLoader

from vc import *
from vc import Utils
from vc.metadata import CreateMetadata
from vc.train import Train
from vc.vc_inference import Inference

app = Flask(__name__)
api = Api(app)

@app.route("/tts", methods=["POST"])        
def tts():
    #numpy array representing the audio data
    print("tts")
    params = request.get_json()
    language_id = int(params['language'])
    text = params['text']
    user_id = params['user']
    
    np_audio = TextAudioLoader(language_id, text).audio
    Inference(np_audio, user_id).inference()
    #save it to BytesIO object(buffer for bytes object)
    #bytes_wav = bytes()
    #byte_io = io.BytesIO(bytes_wav)
    #write(byte_io, 24000, np_audio.audio[::])

    #return wav file
    #return send_file(byte_io, mimetype="audio/x-wav")
    return send_file('vc/output_audio_dir/converted_p400.wav')


@app.route("/put_data", methods=["POST"])
def put_data():
    req = dict(request.form)
    audio = request.form.get("audio")
    print(bytearray(audio.encode()))
    #print(bytearray(audio.encode()))
 
    s = io.BytesIO(audio.encode())
    AudioSegment.from_file(s).export(save_path+'/'+str(text_id)+'.wav', 2, 24000, 1)
    
    data = json.loads(req['data'])

    user_id = data['userId']
    text_id = data['textId']
    save_path = './vc/Data/p' + str(user_id)

    if not os.path.isdir(save_path):
        #data dir 생성
        os.makedirs(save_path, exist_ok=True)
        shutil.copy('./vc/Data/train_list.txt', save_path)
        shutil.copy('./vc/Data/val_list.txt', save_path)

    #data, samplerate = sf.read(io.BytesIO(au))
    #sf.write(save_path + '/' + str(text_id) + '.wav', data, 24000)

    #with open(save_path + '/' + str(text_id) + '.wav', mode='wb') as f:
    #    f.write(bytearray(audio.encode()))

    #input = request.files['audio']
    #input.save(save_path + '/' + str(text_id) + '.wav')

    response = {"textId": text_id}
    return jsonify(response), 200


@app.route("/train", methods=["POST"])
def train():
    params = request.get_json()
    user_id = params['userId']

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

