import tensorflow as tf
import soundfile as sf

import yaml
import numpy as np

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

from TTS.api import TTS


class TextAudioLoader():
    def __init__(self, language_id, text):
        self.language_id = language_id
        self.text = text

        if language_id==1 or language_id==4:
            self.audio = self.coqui()
        else:
            self.audio = self.tensorflow()

    def coqui(self):
        #uk
        if(self.language_id == 1):
            MODEL_NAME="tts_models/en/vctk/vits"
            SPEAKER_ID = 34
        #japanese
        elif(self.language_id == 4):
            MODEL_NAME="tts_models/ja/kokoro/tacotron2-DDC"
            SPEAKER_ID = -1

        tts = TTS(model_name = MODEL_NAME, progress_bar=False, gpu=True)

        #single-speaker model
        if(SPEAKER_ID == -1):
            audio = np.array(tts.tts(self.text), dtype=np.float32)
        #multi-speaker model
        else:
            audio = np.array(tts.tts(self.text, speaker=tts.speakers[SPEAKER_ID]),dtype=np.float32)

        return audio


    def tensorflow(self):
        #load model
        tacotron2, mb_melgan, processor = self.load_model()
        #audio synthesis
        audio = self.do_synthesis(tacotron2, mb_melgan, processor)
        
        return audio


    def load_model(self):
        language_id = self.language_id
        
        if language_id==0:
            tacotron2_path="tts-tacotron2-ljspeech-en"
            melgan_path="tts-mb_melgan-ljspeech-en"
        if language_id==2:
            tacotron2_path="tts-tacotron2-kss-ko"
            melgan_path="tts-mb_melgan-kss-ko"
        if language_id==3:
            tacotron2_path="tts-tacotron2-baker-ch"
            melgan_path="tts-mb_melgan-baker-ch"
        if language_id==5:
            tacotron2_path="tts-tacotron2-synpaflex-fr"
            melgan_path="tts-mb_melgan-synpaflex-fr"
        
        tacotron2 = TFAutoModel.from_pretrained("tensorspeech/"+tacotron2_path, name="tacotron2")
        mb_melgan = TFAutoModel.from_pretrained("tensorspeech/"+melgan_path, name="mb_melgan")
        processor = AutoProcessor.from_pretrained("tensorspeech/"+tacotron2_path)
    
        return tacotron2, mb_melgan, processor


    def do_synthesis(self, text2mel_model, vocoder_model, processor_model):
        input_ids = processor_model.text_to_sequence(self.text)

        # text2mel part
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )

        # vocoder part
        audio = vocoder_model.inference(mel_outputs)[0, :, 0]

        return audio.numpy()
