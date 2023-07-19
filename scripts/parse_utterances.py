import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import argparse
from utils.Ksponspeech_utils import build_utterance_list
import json
from pathlib import Path

parser = argparse.ArgumentParser("Parsing Dataset Utterances to json file")
parser.add_argument("--wav_dir", default ='/home/private_data/KoreanSpeech/test-clean') #Ksponspeech_dir
parser.add_argument("--textgrid_dir", default='../procedure/text_grid/test-clean')
parser.add_argument("--out_file", default='../procedure/parse_output/parse_output.json')
parser.add_argument("--merge_shorter", type=float, default=0.3)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(Path(args.out_file).parent, exist_ok=True)
    utterances = build_utterance_list(args.wav_dir, args.textgrid_dir, merge_shorter=args.merge_shorter)

    with open(args.out_file, "w") as f:
        json.dump(utterances, f, indent=4, ensure_ascii=False)
