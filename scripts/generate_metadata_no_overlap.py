import argparse
import os
import random
import numpy as np
import json
import glob
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import collections


parser = argparse.ArgumentParser("Generating mixtures")
parser.add_argument("--parse_file", default='../procedure/parse_output/parse_output.json')
parser.add_argument("--noise_dir", default='/home/private_data/wham_noise/cv')
parser.add_argument("--out_json", default='../metadata/sparse_2_0/metadata.json')
parser.add_argument("--n_mixtures", default=3000,  type=int)
parser.add_argument("--n_speakers", type=int, nargs='+', default=[2, 3])
parser.add_argument('--ovr_ratio', type=float, default=0.0,
                    help='target overlap amount')
parser.add_argument("--maxlength", type=int, default=15)
parser.add_argument('--random_seed', type=int, default=777,
                    help='random seed')
parser.add_argument("--version", type=int, default=2)

def find_sub_utts_subsets(c_utt, minlen): # c_utt = [{},{},{},{}], 하나의 wav
    valid = []
    for i in range(len(c_utt)):
        sum_till_now = 0
        for j in range(i+1, len(c_utt)):
            sum_till_now += c_utt[j]["stop"] - c_utt[j]["start"]

            if sum_till_now <= minlen:
                valid.append([sum_till_now, [i, j]])

    # select a random utterance from the valid ones
    # selection can be conditioned on the contiguous subarray which goes nearest to minlen # 발화 리스트 내 음성의 최소 길이와 가장 가까운 애로 선택
    if valid:
        valid = sorted(valid, key= lambda x : abs(x[0]-minlen))[0] #key는 -& ~ 0사이 값, 0에 가까울수록 마지막으로 sorting

        start, stop = valid[-1]
        return c_utt[start:stop+1]
    else:
        return [np.random.choice(c_utt)] # all utterances are longer #모든 서브 발화 set이 mindur 보다 길 경우


if __name__ == "__main__":

    args = parser.parse_args()
    VERSION = args.version

    for spk_ in args.n_speakers:
        args.out_json = '../metadata/sparse_{}_0/metadata.json'.format(spk_)

        os.makedirs(Path(args.out_json).parent, exist_ok=True)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        f_name = Path(args.noise_dir).name + f'_max{args.maxlength}s.json'
        if os.path.isfile(f_name):
            print("Loading noise file lists")
            with open(f_name, 'r') as f:
                noises = json.load(f)
        else:
            noises = glob.glob(os.path.join(args.noise_dir, "**/*.wav"), recursive=True)
            prev_len = len(noises)
            noises = [x for x in noises if len(sf.SoundFile(x)) >= args.maxlength*sf.SoundFile(x).samplerate] # noise 선택 : sound file의 길이가 sr x 15 보다 긴 애들만 선정
            print("Number of noise wavs : {}".format(len(noises)))
            print("Discarded : {}".format(prev_len - len(noises))) # 전체 noise에서 maxlength보다 작은 애들은 빼버림
            with open(f_name, 'w') as f:
                json.dump(noises, f) #사용할 noise 선정해서 list에 저장

        with open(args.parse_file, "r") as f:
            utterances = json.load(f)

        all_speakers = list(utterances.keys()) #speaker id

        total_metadata = []
        ct = 0
        mix_n = 0

        while mix_n != args.n_mixtures:

            # recording ids are mix_0000001, mix_0000002, ...
            recid = 'mix_{:07d}'.format(mix_n + 1)

            if spk_ == 2:
                if len(all_speakers) == 1:
                    c_speakers = []
                    for i in range(spk_):
                        c_speakers.append(all_speakers[0])
                else:
                    c_speakers = random.sample(all_speakers, spk_)

            else:
                if len(all_speakers) == 1:
                    c_speakers = []
                    for i in range(spk_):
                        c_speakers.append(all_speakers[0])

                elif len(all_speakers) == 2:
                    NotImplementedError

                else:
                    c_speakers = random.sample(all_speakers, spk_) #비복원추출

            metadata = []
            maxlength = -1

            while True:
                tmp_ = []
                utts = [random.choice(utterances[spk]) for spk in c_speakers] # speech random sampling

                for i in range(len(utts)):
                    tmp_.append(utts[i][0]['utt_id'])

                if len(set(tmp_)) < spk_: #1. 같은 speech sampling 방지
                    print('same speech exists! sampling again.')
                    continue

                mindur_spk = np.inf
                du_list = []
                sub_key = 0

                for spk_indx in range(len(utts)):
                    if sub_key == 1:
                        print('too short sub utterance length for making mixture. sampling again.')
                        break
                    tmp = 0

                    for sub_utt in utts[spk_indx]:
                        if sub_utt["stop"] - sub_utt["start"] < 0.1: #2. sub 발화가 너무 짧을 경우 다시 sampling, make_mixture.py의 block size보다 길어야함.
                            sub_key = 1
                            break

                        tmp += sub_utt["stop"] - sub_utt["start"]

                    if tmp <= 3: #3. min duration 길이가 3이하일 경우 다시 sampling
                        break
                    du_list.append(tmp)

                if len(du_list) == len(utts):
                    mindur_spk = min(du_list) #
                    break

            if spk_ == 2:
                if len(all_speakers) == 1:
                    c_speakers[1] += '_dummy'
                    for i in range(len(utts[1])):
                        utts[1][i]['spk_id'] = c_speakers[1]

            else:
                if len(all_speakers) == 1:
                    c_speakers[1] += '_dumm1'
                    c_speakers[2] += '_dumm2'
                    for i in range(len(utts[1])):
                        utts[1][i]['spk_id'] = c_speakers[1]
                    for j in range(len(utts[2])):
                        utts[2][j]['spk_id'] = c_speakers[2]

                elif len(all_speakers) == 2:
                    NotImplementedError

            # having minimum duration we keep adding utterances from one speaker till we have minimum duration
            kept = []
            for spk_indx in range(len(utts)):

                tmp = find_sub_utts_subsets(utts[spk_indx], mindur_spk)
                kept.append(tmp[::-1]) # we use pop after thus we reverse here

            utts = kept
            for i in range(len(utts)):
                for j in range(len(utts[i])):
                    ct +=1

            lasts = {}
            for i in c_speakers:
                lasts[i] = [0, 0]


            overlap_stat = 0
            tot = 0
            sub_utt_num = 0

            #print('utts :', utts) # [ [{}, {}] , [{}, {}] ]

            while any(utts): # till we have utterances
                if tot == 0:
                    spk_indx = 0
                    prev_spk_indx = 1
                else:
                    prev_spk_indx = spk_indx
                    spk_indx = random.choice([x for x in range(len(c_speakers)) if x != prev_spk_indx]) #서브 발화를 번갈아가면서 빼내는 부분

                # if number of sub_utts for this speaker is greater than n sub utts of all other
                # we can afford to not overlap this utterance on the left
                try:
                    sub_utt = utts[spk_indx].pop()
                except:
                    continue # no more utterances for this speaker

                c_spk = c_speakers[spk_indx]
                prev_spk = c_speakers[prev_spk_indx]
                #print('prev_spk_indx :', prev_spk_indx)
                #print('spk_indx :', spk_indx)

                if lasts[prev_spk][-1] != 0: #직전에 사용했던 utterance가 있을 시
                    print(lasts)
                    # not first utterance
                    if VERSION == 1:
                        stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0]) #c_spk가 아닌 타 spk 중에서 마지막 lasets[x][-1]값이 가장 작은애들
                        start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                        lastlen = stop - start
                        if args.ovr_ratio != 0:
                            raise NotImplemented
                        else:
                            # This should always be the stop of the previous speaker.
                            it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05 # 다음 sub utterance의 start 시간 결정
                    elif VERSION == 2:
                        start, stop = lasts[prev_spk]
                        lastlen = stop - start
                        if args.ovr_ratio != 0:
                            raise NotImplemented
                        else:
                            # This should always be the stop of the previous speaker.
                            it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05
                    elif VERSION == 3:
                        c_length = sub_utt["stop"] - sub_utt["start"]
                        stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0])
                        start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                        lastlen = stop - start
                        if args.ovr_ratio != 0:
                            raise NotImplemented

                        else:
                            # This should always be the stop of the previous speaker.
                            it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05
                    else:
                        raise ValueError

                    # if offset == 0# no overlap maybe we can use pauses between utterances
                else:
                    it = np.random.uniform(0.2, 0.5)  # first utterance

                maxlength = max(maxlength, it + (sub_utt["stop"] - sub_utt["start"]))

                if '_dummy' in sub_utt['spk_id'] or '_dumm1' in sub_utt['spk_id'] or '_dumm2' in sub_utt['spk_id']:

                    #print('speaker_dummy label will be modified to {}'.format(c_speakers[0]))
                    sub_utt['spk_id'] = sub_utt['spk_id'][:-6]

                c_meta = {"file": "/".join(sub_utt["file"].split("/")[-3:])[:-3] + 'wav', "words": sub_utt["words"],
                          "spk_id": sub_utt["spk_id"],
                          "chapter_id": sub_utt["chapter_id"], "utt_id": sub_utt["utt_id"],
                          "start": np.round(it, 3),
                          "stop": np.round(it + (sub_utt["stop"] - sub_utt["start"]), 3),
                          "orig_start": sub_utt["start"],
                          "orig_stop": sub_utt["stop"], "lvl":  np.random.uniform( -33, -25),
                          "source": "s{}".format(spk_indx + 1), "sub_utt_num": sub_utt_num }
                metadata.append(c_meta)
                #print(metadata)
                lasts[c_spk][0] = c_meta["start"]
                lasts[c_spk][1] = c_meta["stop"]  # can't overlap with itself
                #print('lasts :', lasts)
                tot += c_meta["stop"] - c_meta["start"]
                sub_utt_num += 1

            ## noise ##
            maxlength += np.random.uniform(0.2, 0.5) # ASR purposes we add some silence at end
            noise = np.random.choice(noises)

            # if noise file is more than maxlength then we take a random window

            if len(sf.SoundFile(noise)) - int(maxlength * sf.SoundFile(noise).samplerate) <= 0 :
                print("TEST ONLY too long utterance skipping utt")
                continue
            offset = random.randint(0, len(sf.SoundFile(noise)) - int(maxlength*sf.SoundFile(noise).samplerate)) #noise start 결정

            c_lvl = np.random.uniform(-38, -30) #np.clip(first_lvl - random.normalvariate(3.47, 4), -40, 0)
            #print(len(sf.SoundFile(noise)))
            metadata.append({"file": noise.split("/")[-1],
                             "start": 0,
                             "stop": maxlength, "orig_start": np.round(offset/sf.SoundFile(noise).samplerate, 3),
                             "orig_stop": np.round(offset/sf.SoundFile(noise).samplerate + maxlength, 3),
                             "lvl": c_lvl, "source": "noise", "channel": random.randint(0,1)})

            mixture_metadata = {"mixture_name": recid}
            for elem in metadata:
                if elem["source"] not in mixture_metadata.keys():
                    mixture_metadata[elem["source"]] = [elem]
                else:
                    mixture_metadata[elem["source"]].append(elem)

            total_metadata.append(mixture_metadata)
            mix_n += 1
            print('creating {} of {} mixtures ...'.format(mix_n, args.n_mixtures))

        print('noise number :', len(noises))
        print('total sub utterance number : ', ct)
        with open(os.path.join(args.out_json), "w") as f:
            json.dump(total_metadata, f, indent=4, ensure_ascii=False)


























