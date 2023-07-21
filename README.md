# SparseKMix
- KMix로 train한 single channel speech separation model evaluation 데이터셋 생성 SW입니다.

- 실제 중첩음과 유사한 인공 중첩음 데이터로 모델을 평가하기 위해 제작되었습니다.

- 생성에 필요한 metadata를 포함한 레포지토리 입니다.

  

## How To Run

```bash
git clone https://github.com/sogang-isds/SparseKMix.git
cd SparseKMix
pip install -r requirements.txt

./create_sparse.sh # path/to/SparseKMix/storage_dir에 생성됨
```



## How to make metadata(not essential)



### 0. MFA Installation

- metadata 생성을 위해서 발화-음소 정렬기인 MFA(Montreal Forced Aligner)를 사용합니다.

(Github) https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner

(Documents)  https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/corpus_structure.html

```bash
git clone https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
cd path/to/Montreal-Forced-Aligner #경로 이동

conda env create -n mfa-dev -f path/to/environment.yml #requirements 설치
conda activate mfa-dev

pip install -e .[dev] #dependency 설치
```

- 설치 이후 **mfa version** 입력을 통해 설치가 제대로 되었는지 확인이 가능합니다.
- mfa관련 모든 작업은 conda 가상환경에서 실행해야합니다.



### 1. Download the pre-trained model and dictionary

```bash
#한국어 pretrained 음향모델 및 dictionary : korean_mfa
mfa model download acoustic (model_name)
mfa model download dictionary (dictionary name)
```

- download한 file들은 /home/(user_id)/Documents/MFA에 저장됩니다.

### 2. Create Textgrid

```bash
mfa validate /home/private_data/KoreanSpeech/test-clean(코퍼스 경로) korean_mfa(발음사전)
korean_mfa(음향모델)

mfa align /home/private_data/KoreanSpeech/test-clean(코퍼스 경로) korean_mfa(발음사전) korean_mfa(음향
모델) /home/(user_id)/path/to/SparseKMix/procedure/text_grid --clean
```

### 3. Create metadata

이후 **SparseKMix/scripts** 경로로 이동한 뒤 아래 명령어를 실행시키면 metadata 생성이 완료됩니다.

```bash
pip install -r requirements.txt
cd scripts

python parse_utterances.py
python generate_metadata_no_overlap.py
python generate_metadata_overlap.py
```

