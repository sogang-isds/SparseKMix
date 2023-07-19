import os
import glob
from pathlib import Path
from .textgrid_utils import build_hashtable_textgrid, get_textgrid_sa
import soundfile as sf
import numpy as np


def hash_librispeech(librispeech_traintest):

    hashtab = {}
    utterances = glob.glob(os.path.join(librispeech_traintest, "**/*.wav"), recursive=True)
    for utt in utterances:
        id = Path(utt).parent.parent
        hashtab[id] = utt
    return hashtab

def ema_energy(x, alpha=0.99):

    out = np.sum(x[0]**2)
    for i in range(1, len(x)):
        out = (1-alpha)*np.sum(x[i]**2) + alpha*out
    return out

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def estimate_snr(speech, file):

    def get_energy(signal): # probably windowed estimation is better but we take min afterwards and

        return np.sum(signal**2) #해당 duration 길이 내 존재하는 amplitude값을 제곱해서 모두 더한 값

    audio, fs = sf.read(file) #numpy형식 audio data, sampling rate 리턴
    audio = audio - np.mean(audio) #노말라이징
    s, e = [int(x*fs) for x in speech[0]] #start duration, end duration
    speech_lvl = [get_energy(audio[s:e])]
    noise_lvl = [get_energy(audio[0: s])] #각 음성의 처음 부분 silence 구간을 첫 noise level로 봄

    for i in range(1, len(speech)):
        speech_lvl.append(get_energy(audio[int(speech[i][0]*fs):int(speech[i][-1]*fs)])) #발화 구간
        noise_lvl.append(get_energy(audio[int(speech[i-1][-1]*fs): int(speech[i][0]*fs)])) #slience 구간

    noise_lvl.append(get_energy(audio[int(speech[-1][-1]*fs):])) #마지막 silence 구간

    noise_lvl = min(noise_lvl) # we take min to avoid breathing
    speech_lvl = min(speech_lvl)

    return speech_lvl / (noise_lvl + 1e-8) # noise_lvl == 0일 경우 방지



def build_utterance_list(wav_dir, textgrid_dir, merge_shorter=0.15, fs=16000, use_chapter=False):


    hashgrid = build_hashtable_textgrid(textgrid_dir) #{textgird 파일이름 : 경로 ...} 반환
    audiofiles = glob.glob(os.path.join(wav_dir, "**/*.wav"), recursive=True) #flac => wav

    utterances = {} # { spk_id : [sub_utter] }
    tot_missing = 0
    snrs = []

    for f in audiofiles:

        filename = Path(f).stem #확장자 없어짐
        if filename not in hashgrid.keys():
            print("Missing Alignment file for : {}".format(f))
            tot_missing += 1
            continue
        speech, words = get_textgrid_sa(hashgrid[filename], merge_shorter) #하나의 음성 파일에 대한 sub utterance의 start, end time 및 단어

        """
        missing alignment 원인 파악 필요
        speech = 이중리스트, [[start_time[0], end_time[0], [str_time[1], end_time[1]], ,,,, []]
        get_textgrid_sa = 각 음성마다 sub utterances를 생성. 이때 start time, end time 정보가 함께 추출 됨. 
        merge_shorter 비율에 대한 issue
        Librispeech를 들어보면 간 서브 발화(하나의 문장) 내 단어들의 발화 간격이 짧아서 0.15로 결정한 것 같음. 
        Ksponspeech는 긴 편. 일단 0.3으로 샘플 mixture를 생성했지만 논의 필요.       
        snr이 낮은 wav들어보니 명확하게 발화가 잘되는 경우가 대부분임.
        chapter별이 아닌 음성파일별로 snr 계산해서 제거하는 방식 코드 추가
        221028 block size issue 해결 필요
        221101 block size는 wav 길이에 적용되고 fs와 곱해져서 프로세스 처리에 들어가는 음성 단위를 결정. 0.1 => 0.1초 간격
                서브 발화가 0.1초보다 작은 case가 발생하면 노말라이징이 안되는 특징이 있음. 서브발화 길이 결정의 문제
        """


        spk_id = Path(f).parent.parent.stem

        sub_utterances = []
        # get all segments for this speaker
        if not speech:
            raise EnvironmentError("something is wrong with alignments or parsing, all librispeech files have speech")

        snr = estimate_snr(speech, f) #한 파일마다 snr 계산
        snrs.append([f, snr])

        for i in range(len(speech)):
            start, stop = speech[i]
            #start = #int(start*fs)
            #stop = int(stop*fs)
            tmp = {"textgrid": hashgrid[filename], "file": f, "start": start, "stop": stop, "words": words[i],
                   "spk_id": spk_id, "chapter_id": Path(f).parent.stem, "utt_id": Path(f).stem}
            sub_utterances.append(tmp)

        if spk_id not in utterances.keys():
            utterances[spk_id] = [sub_utterances]
        else:
            utterances[spk_id].append(sub_utterances) #speaker id별로 sub utterance 넣어주는 부분
            # utterances의 key는 spk_id, value는 sub_utterances인데, value내 서브 발화는 여러음성 파일이 모아진 것 => spk_id : [[sub_utter1], [sub_utter2].... [sub_utter n]]

    # here we filter utterances based on SNR we sort the snrs list and if a chapter has more than 10 entries with snr lower than
    # 10 we remove that chapter
    snrs = sorted(snrs, key = lambda x : x[-1]) #snr이 작은 file 부터 오름차순으로 정렬

    low_snr_list =[]

    if use_chapter:

        chapters = {}
        for x in snrs:
            file, snr = x
            chapter = Path(file).parent.stem #wav의 상위 폴더명만 남김
            if chapter not in chapters.keys():
                chapters[chapter] = [0, 0]
            chapters[chapter][0]  += 1
            if snr < 50:

                low_snr_list.append(file)
                chapters[chapter][-1] += 1 #chapter에 속한 wav 수 : [0], chapter에 속한 wav 중 snr이 일정 기준 이하인 음성 수 : [1]

        print('low_snr_count :', len(low_snr_list))

        # normalize
        for k in chapters.keys():
            chapters[k] = chapters[k][-1] / chapters[k][0]

        b_tot_utterances = sum([len(utterances[k]) for k in utterances.keys()]) #총 sub 발화 수
        new = {}
        for spk in utterances.keys():
            new[spk] = []
            for utt in utterances[spk]:
                if chapters[utt[0]["chapter_id"]] >= 0.1: #챕터 내 snr이 낮은 wav 비율이 10% 이상인 경우 해당 챕터 전부 제외
                    continue
                else:
                    new[spk].append(utt)
            if len(new[spk]) == 0:
                continue
        utterances = new

    else:
        for x in snrs:
            file, snr = x
            if snr < 50: #SNR 수준은 음성신호 처리 기준을 따라야 할지 디지털 신호 처리기준을 따라야할지 결정 (50, 15)
                low_snr_list.append(Path(file).stem)

        b_tot_utterances = sum([len(utterances[k]) for k in utterances.keys()])  # 총 sub 발화 수
        new = {}
        for spk in utterances.keys():
            new[spk] = []
            for utt in utterances[spk]:
                if utt[0]['utt_id'] in low_snr_list:
                    continue
                else:
                    new[spk].append(utt)
            if len(new[spk]) == 0:
                continue

        utterances = new

    a_tot_utterances = sum([len(utterances[k]) for k in utterances.keys()])
    print('total audio :', len(audiofiles))
    print('------------------ except missing alignment number : {} ------------------ '.format(tot_missing))
    print('before_snr_processing wav number :', b_tot_utterances)
    print('--------------------------- low_snr_count : {} -------------------------- '.format(len(low_snr_list)))
    print('after_snr_processing wav number :', a_tot_utterances)

    print("Discarded {} over {} files because of low SNR".format(b_tot_utterances - \
                                                               sum([len(utterances[k]) for k in utterances.keys()]), b_tot_utterances))

    return utterances

if __name__ == "__main__":
    build_utterance_list("/media/sam/Data/LibriSpeech/test-clean/", "/home/sam/Downloads/librispeech_alignments/test-clean/")
