#!/bin/bash
wav_dir=/home/private_data/KoreanSpeech/test-clean # path to KsponSpeech test-clean
noise_dir=/home/private_data/wham_noise/cv # path to test WHAM noises
stage=0
fs=16000
all_overlap="0 0.2 0.4 0.6 0.8 1.0"

set -e

if [[ $stage -le 0 ]]; then
    for fs in 16000; do
      for n_speakers in 2 3; do
        for ovr_ratio in $all_overlap; do
          echo "Making mixtures for ${n_speakers} speakers and overlap ${ovr_ratio}"
          python scripts/make_mixtures.py --wav_dir $wav_dir --noise_dir $noise_dir --rate $fs
          done
      done
    done
fi
