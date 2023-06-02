import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import argparse
from pathlib import Path
from scipy.io.wavfile import write, read

from vc.Utils.ASR.models import ASRCNN
from vc.Utils.JDC.model import JDCNet
from vc.models import Generator, MappingNetwork, StyleEncoder

import soundfile as sf
import sys
from parallel_wavegan.utils import load_model

import os

# Source: http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is18/en_speaker_used.txt
# Source: https://github.com/jjery2243542/voice_conversion
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

speakers = [225,228,229,230,231,233,236,239,1,244,226,227,232,243,254,256,258,5,270,273]

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

output_path = "vc/output_audio_dir"
# model_path = "../StarGANv2-VC/Models/VCTK20/epoch_00148.pth"


class Inference:

    def __init__(self, input_audio, user_id):
        self.input_audio = input_audio
        self.user_id = user_id
        if user_id == 3:  #민정이 아이디면 나 넣고 민정이 넣음
            speakers.append(2)
        speakers.append(user_id)
        if user_id == 2:  #내 아이디이면 나 넣고 민정이 넣음
            speakers.append(3)
       # torch.backends.cudnn.enabled = False

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def build_model(self, model_params={}):
        args = Munch(model_params)
        generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
        mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
        style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

        nets_ema = Munch(generator=generator,
                         mapping_network=mapping_network,
                         style_encoder=style_encoder)

        return nets_ema

    def compute_style(self, speaker_dicts):
        global starganv2
        reference_embeddings = {}
        for key, (path, speaker) in speaker_dicts.items():
            if path == "":
                label = torch.LongTensor([speaker]).to('cuda')
                latent_dim = starganv2.mapping_network.shared[0].in_features
                ref = starganv2.mapping_network(
                    torch.randn(1, latent_dim).to('cuda'), label)
            else:
                wave, sr = librosa.load(path, sr=24000)
                audio, index = librosa.effects.trim(wave, top_db=30)
                if sr != 24000:
                    wave = librosa.resample(wave, sr, 24000)
                mel_tensor = self.preprocess(wave).to('cuda')

                with torch.no_grad():
                    label = torch.LongTensor([speaker])
                    ref = starganv2.style_encoder(
                        mel_tensor.unsqueeze(1), label)
            reference_embeddings[key] = (ref, label)

        return reference_embeddings

    def inference(self):

        # load F0 model
        F0_model = JDCNet(num_class=1, seq_len=192)
        params = torch.load("vc/Utils/JDC/bst.t7")['net']
        F0_model.load_state_dict(params)
        _ = F0_model.eval()
        F0_model = F0_model.to('cuda')


        # load vocoder
        vocoder = load_model(
            "vc/Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
        vocoder.remove_weight_norm()
        _ = vocoder.eval()


        # load starganv2
        global starganv2
        model_path = 'vc/Models/p' + str(self.user_id) + '/epoch_00148.pth'
        config_path = 'vc/Configs/p' + str(self.user_id)  + '/config.yml'

        with open(config_path) as f:
            starganv2_config = yaml.safe_load(f)
        starganv2 = self.build_model(
            model_params=starganv2_config["model_params"])
        params = torch.load(model_path, map_location='cpu')
        params = params['model_ema']
        _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
        _ = [starganv2[key].eval() for key in starganv2]
        starganv2.style_encoder = starganv2.style_encoder.to('cuda')
        starganv2.mapping_network = starganv2.mapping_network.to('cuda')
        starganv2.generator = starganv2.generator.to('cuda')


        # load input wave
        selected_speakers = [self.user_id] #TODO: 243, 244, 236, 233, 230, 228, self.user_id
        #k = random.choice(selected_speakers)
        #audio, source_sr = librosa.load(self.input_audio, sr=24000)
        audio = self.input_audio
        #audio = audio / np.max(np.abs(audio))
        #audio.dtype = np.float32

        # with reference, using style encoder
        speaker_dicts = {}
        for s in selected_speakers:
            k = s
            speaker_dicts['p' + str(s)] = ('./vc/Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))

        reference_embeddings = self.compute_style(speaker_dicts)

        # conversion
        import time
        start = time.time()

        source = self.preprocess(audio).to('cuda:0')  #test 전: audio
        keys = []
        converted_samples = {}
        reconstructed_samples = {}
        converted_mels = {}

        for key, (ref, _) in reference_embeddings.items():
            with torch.no_grad():
                f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
                out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

                c = out.transpose(-1, -2).squeeze().to('cuda')
                y_out = vocoder.inference(c)
                y_out = y_out.view(-1).cpu()

                if key not in speaker_dicts or speaker_dicts[key][0] == "":
                    recon = None
                else:
                    wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                    mel = self.preprocess(wave)
                    c = mel.transpose(-1, -2).squeeze().to('cuda')
                    recon = vocoder.inference(c)
                    recon = recon.view(-1).cpu().numpy()

            converted_samples[key] = y_out.numpy()
            reconstructed_samples[key] = recon

            converted_mels[key] = out

            keys.append(key)
        end = time.time()
        print('total processing time: %.3f sec' % (end - start))

        #return converted_samples['p'+str(self.user_id)]

        for key, wave in converted_samples.items():
            sf.write(output_path+"/converted_"+key+".wav", wave, 24000)
            if reconstructed_samples[key] is not None:
                sf.write(output_path+"/reference_"+key+".wav", reconstructed_samples[key], 24000)
        
        return converted_samples['p'+str(self.user_id)]

    #sf.write(output_path+"/original.wav", wav_path, 24000)


# if __name__=="__main__":
#    parser = argparse.ArgumentParser(description="Generate a voice conversion audio file from input audio file.")
#    parser.add_argument(
#        "model_path",
#        metavar="model_path",
#        help="path to the model directory.",
#        type= Path,
#    )
    # parser.add_argument(
    #    "input_dir",
    #    metavar="input_dir",
    #    help="path to the source directory.",
    #    type= Path,
    # )
#    parser.add_argument(
#        "output_dir",
#        metavar="output_dir",
#        help="path to the output directory.",
#        type= Path,
#    )
#    args = parser.parse_args()
#    main(args)

# python vc_inference.py {input path}
