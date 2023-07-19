import argparse
import json
import numpy as np
import random
from pathlib import Path
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("generate overlap mixtures from non-overlap ones")
parser.add_argument("--no_ov_metadata", default='../metadata/sparse_2_0/metadata.json')
parser.add_argument("--out_json", default='../metadata/sparse_2_0/metadata.json')
parser.add_argument('--ovr_ratio', type=float, nargs='+', default=[0.2,0.4,0.6,0.8,1.0],
                    help='target overlap amount')
parser.add_argument("--version", type=int, default=2)
parser.add_argument("--n_speakers", type=int, nargs='+', default=[2, 3])
args = parser.parse_args()
VERSION = args.version

for spk_ in args.n_speakers:
    for index_ in args.ovr_ratio:

        args.out_json = '../metadata/sparse_{}_{}/metadata.json'.format(spk_, index_)
        args.no_ov_metadata = '../metadata/sparse_{}_0/metadata.json'.format(spk_)
        assert index_ != 0

        with open(args.no_ov_metadata, "r") as f:
            no_ov = json.load(f)


        total_metadata = []
        for mixture in tqdm(no_ov):

            all_speaker = []
            for j in mixture:
                if j.startswith("s"):
                    all_speaker.append(mixture[j][0]['spk_id'])

            #print(set(all_speaker))
            # for each mixture
            # we sort all sub_utts by their sub_utt number
            sub_utts = []

            if spk_ == 2:
                if len(set(all_speaker)) == 1:
                    for k in mixture:
                        if k == 's2':
                            for i in range(len(mixture[k])):
                                mixture[k][i]['spk_id'] += '_dummy'

                        if k.startswith("s"):  # is a source
                            for sub in mixture[k]:
                                sub_utts.append(sub)
            else:

                if len(set(all_speaker)) == 1:
                    for k in mixture:
                        if k == 's2':
                            for i in range(len(mixture[k])):
                                mixture[k][i]['spk_id'] += '_dumm1'
                        elif k == 's3':
                            for i in range(len(mixture[k])):
                                mixture[k][i]['spk_id'] += '_dumm2'

                elif len(set(all_speaker)) == 2:
                    raise NotImplementedError

                for k in mixture:
                    if k.startswith("s"):  # is a source
                        for sub in mixture[k]:
                            sub_utts.append(sub)

            sub_utts = sorted(sub_utts, key= lambda x : x["sub_utt_num"])
            #print(sub_utts)
            c_speakers = [mixture[k][0]["spk_id"] for k in mixture.keys() if k.startswith("s")]

            lasts = {}  # collections.deque(maxlen=len(c_speakers)) # circular buffer which contains
            for i in c_speakers:
                lasts[i] = [0, 0]

            overlap_stat = 0
            tot = 0
            metadata = []
            maxlength = -1
            prev_spk = c_speakers[0]

            for sub_utt in sub_utts:

                if tot == 0:
                    c_spk = sub_utt["spk_id"]
                    prev_spk = c_spk
                else:
                    prev_spk = c_spk
                    c_spk = sub_utt["spk_id"]
                print('last : ', lasts)
                if lasts[prev_spk][-1] != 0:
                    # not first utterance
                    if VERSION == 1:

                        stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0])
                        print('stop : ',stop)
                        start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                        print('start : ',start)
                        lastlen = stop - start
                        print('lastlen : ',lastlen)
                        offset = lastlen * index_
                        print('offset :',offset)
                        it = max(stop - offset, lasts[c_spk][-1] + 0.05)
                        print('stop-offset :', stop - offset)
                        print('lasts[c_spk] :', lasts[c_spk][-1] + 0.05)
                        overlap_stat += lasts[prev_spk][-1] - it

                    elif VERSION == 2:
                        start, stop = lasts[prev_spk]
                        lastlen = stop - start

                        offset = lastlen * index_
                        it = max(stop - offset, lasts[c_spk][-1] + 0.05)
                        overlap_stat += lasts[prev_spk][-1] - it

                    elif VERSION == 3:
                        c_length = sub_utt["stop"] - sub_utt["start"]
                        stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0])
                        start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                        lastlen = stop - start

                        offset = lastlen * index_
                        # basically instead of adding it up on the right we include also the option to add the segment on the left
                        # if of course will start > 0
                        if (start + offset) - c_length > 0.1:  #
                            offset = random.choice([stop - offset, (start + offset) - c_length])
                        else:
                            offset = stop - offset
                        it = max(offset, lasts[c_spk][-1] + 0.05)
                        overlap_stat += lasts[prev_spk][-1] - it

                    else:
                        raise ValueError

                    # if offset == 0# no overlap maybe we can use pauses between utterances
                else:
                    it = sub_utt["start"]

                maxlength = max(maxlength, it + (sub_utt["stop"] - sub_utt["start"]))

                tmp_id = ""
                if '_dummy' in sub_utt['spk_id'] or '_dumm1' in sub_utt['spk_id'] or '_dumm2' in sub_utt['spk_id']:

                    sub_utt['spk_id'] = sub_utt['spk_id'][:-6]
                    tmp_id = sub_utt['spk_id'][-6:]

                c_meta = {"file": sub_utt["file"], "words": sub_utt["words"],
                          "spk_id": sub_utt["spk_id"],
                          "chapter_id": sub_utt["chapter_id"], "utt_id": sub_utt["utt_id"],
                          "start": np.round(it, 3),
                          "stop": np.round(it + (sub_utt["stop"] - sub_utt["start"]), 3),
                          "orig_start": sub_utt["orig_start"],
                          "orig_stop": sub_utt["orig_stop"], "lvl": sub_utt["lvl"],
                          "source": sub_utt["source"], "sub_utt_num": sub_utt["sub_utt_num"]}
                metadata.append(c_meta)
                lasts[c_spk][0] = c_meta["start"]
                lasts[c_spk][1] = c_meta["stop"]  # can't overlap with itself
                sub_utt['spk_id'] = sub_utt['spk_id'] + tmp_id
                tot += c_meta["stop"] - c_meta["start"]

                ## noise ##
            maxlength += np.random.uniform(0.2, 0.5)  # ASR purposes we add some silence at end
            noise = mixture["noise"][0]
            # if noisefile is more than maxlength then we take a random window

            #offset = random.randint(0, len(sf.SoundFile(noise)) - int(maxlength * sf.SoundFile(noise).samplerate))
            metadata.append({"file": noise["file"],
                             "start": 0,
                             "stop": maxlength, "orig_start": noise["orig_start"],
                             "orig_stop": noise["orig_start"] + maxlength,
                             "lvl": noise["lvl"], "source": "noise", "channel":noise["channel"]})

            mixture_metadata = {"mixture_name": mixture["mixture_name"]}
            for elem in metadata:
                if elem["source"] not in mixture_metadata.keys():
                    mixture_metadata[elem["source"]] = [elem]
                else:
                    mixture_metadata[elem["source"]].append(elem)

            total_metadata.append(mixture_metadata)

        os.makedirs(Path(args.out_json).parent, exist_ok=True)
        with open(os.path.join(args.out_json), "w") as f:
            json.dump(total_metadata, f, indent=4, ensure_ascii=False)