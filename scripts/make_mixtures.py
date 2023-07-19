import argparse
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
import pyloudnorm
from scipy.signal import resample_poly

parser = argparse.ArgumentParser()
parser.add_argument("--json", default ='../metadata/sparse_2_0/metadata.json')
parser.add_argument("--wav_dir", default='/home/private_data/KoreanSpeech/test-clean')
parser.add_argument('--out_dir', default= '../storage_dir/sparse_2_0',help='output data dir of mixture')
parser.add_argument("--noise_dir", type=str, default="/home/private_data/wham_noise/cv")
parser.add_argument('--rate', type=int, default=16000,
                    help='sampling rate')
parser.add_argument("--n_speakers", type=int, nargs='+', default=[2, 3])
parser.add_argument('--ovr_ratio', type=float, nargs='+', default=[0, 0.2,0.4,0.6,0.8,1.0],
                    help='target overlap amount')
args = parser.parse_args()


def resample_and_norm(signal, orig, target, lvl):
    # print('signal :', len(signal))
    # print('orig :', orig)
    # print('target :', target)
    # print('lvl :', lvl)
    if orig != target:
        signal = resample_poly(signal, target, orig)

    # fx = (AudioEffectsChain().custom("norm {}".format(lvl)))
    # signal = fx(signal)

    meter = pyloudnorm.Meter(target, block_size=0.01)  # block size issue
    """
    sub utterance의 길이가 block_size * sr 보다 작을 경우 valid한 음성으로 보지 않음
    block size는 서브 발화의 최소 길이를 결정하는 것 과 같음. 주어진 sr * 

    """

    loudness = meter.integrated_loudness(signal)  # 소리 크기 측정

    signal = pyloudnorm.normalize.loudness(signal, loudness, lvl)  # normalize 시키기

    return signal

for spk_ in args.n_speakers:
    for index_ in args.ovr_ratio:
        args.json = '../metadata/sparse_{}_{}/metadata.json'.format(spk_ , index_)
        args.out_dir = '../storage_dir/sparse_{}_{}'.format(spk_, index_)

        if not args.noise_dir:
            print("Generating only clean version")

        with open(args.json, "r") as f:
            total_meta = json.load(f)

        #metadata의 경우 wav file과 s1, s2 정보가 저장되어있음.
        #mixture를 만들 때 각각 wav_file의 길이를 max_length로 맞춰주기 위해서 앞과 뒤에 padding 줌



        ct = 0
        for mix in tqdm(total_meta): #여기서 mix는 하나의 파일을 의미 ex : mix_00000001.wav

            filename = mix["mixture_name"]

            sources_list = [x for x in mix.keys() if x != "mixture_name"]

            sources = {}
            maxlength = 0

            for source in sources_list: #sources_list = [s1, s2, noise]
                # read file optional resample it
                source_utts = []

                for utt in mix[source]:
                    if utt["source"] != "noise":
                        utt["file"] = os.path.join(args.wav_dir, utt["file"])
                    else:
                        if args.noise_dir:
                            utt["file"] = os.path.join(args.noise_dir, utt["file"])
                        else:
                            continue

                    utt_fs = sf.SoundFile(utt["file"]).samplerate
                    #print('fs확인 :', utt_fs)
                    #print('rate확인 :', args.rate)  block size 때문에 audio length는 충분히 길어야 함.

                    audio, fs = sf.read(utt["file"], start=int(utt["orig_start"]*utt_fs), #start는 duration 시작지점
                                    stop=int(utt["orig_stop"]*utt_fs))

                    #assert len(audio.shape) == 1, "we currently not support multichannel"
                    if len(audio.shape) > 1: #multichannel의 경우, noise가 해당 됨. mutl channel 분리해서 들어보니 다 같음

                        audio = audio[:, utt["channel"]] #TODO

                    audio = audio - np.mean(audio)

                    audio = resample_and_norm(audio, fs, args.rate, utt["lvl"])

                    audio = np.pad(audio, (int(utt["start"]*args.rate), 0), "constant") # pad the beginning 왼쪽으로만 padding을 줌
                    #print('패딩 결과', audio)
                    source_utts.append(audio)
                    maxlength = max(len(audio), maxlength) #여기서 audio는 왼쪽 pad가 들어간 상태

                    #s1, s2, noise에 속한 모든 utterance들에 np.mean(),resample_and_norm, 앞부분 padding을 주고 maxlen구함
                    #이때 max_len
                sources[source] = source_utts #sources dictionary 내에 s1,s2,noise를 key로 주고 각 subset utterance가 담긴 list를 value로 줌

            #print('noise :', len(sources['noise']))
            #print('s2 :' ,len(sources['s2']))

            # pad everything to same length
            for s in sources.keys(): # s1, s2, noise
                for i in range(len(sources[s])):

                    tmp = sources[s][i] #s1, s2, noise의 list 내 인덱스로 접근
                    sources[s][i] = np.pad(tmp,  (0, maxlength-len(tmp)), 'constant') #오른쪽에 max_length-len(tmp)만큼 padding 줌
                    #print('key :',s, 'num_utter :', i)
                    #print('max_length :', maxlength)
                    #print('tmp_len :', len(tmp)) #noise의 길이가  최대 길이가 됨.

            # mix n sum
            tot_mixture = None
            for indx, s in enumerate(sources.keys()):
                if s == "noise": #먼저 clean data들에 대해서만 mix함
                    continue

                source_mix = np.sum(sources[s], 0) #s1, s2 별로 먼저 합치기, np.sum으로 list 내 np.ndarray sum
                # print('output : ', source_mix[48000:48005])

                os.makedirs(os.path.join(args.out_dir, s), exist_ok=True) #s1, s2 mix된 data 따로 만듦
                sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
                if indx == 0:
                    tot_mixture = source_mix #s1, s2끼리 합쳐주기
                else:

                    tot_mixture += source_mix


            os.makedirs(os.path.join(args.out_dir, "mix_clean"), exist_ok=True) #mix clean data 따로 만듦
            sf.write(os.path.join(args.out_dir, "mix_clean", filename + ".wav"), tot_mixture, args.rate)

            if args.noise_dir:

                s = "noise"

                source_mix = np.sum(sources[s], 0) #noise data 따로 만듦(multi-channel 경우 고려해서 작성)

                os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)
                sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
                tot_mixture += source_mix #s1, s2, noise 다 합쳐주는 부분
                os.makedirs(os.path.join(args.out_dir, "mix_noisy"), exist_ok=True)
                sf.write(os.path.join(args.out_dir, "mix_noisy", filename + ".wav"), tot_mixture, args.rate)























