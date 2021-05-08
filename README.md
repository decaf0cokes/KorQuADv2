# Machine Reading Comprehension(기계독해): KorQuADv2

[KorQuADv2](https://korquad.github.io/)는 한국어 Machine Reading Comprehension(기계독해) 데이터셋으로 KorQuADv1에 비해 문서의 길이가 길고, HTML Tag가 등장한다는 특징이 있다.

Transformer Encoder 기반의 모델들(BERT 등)은 일반적으로 Input Sequence의 Max Length가 고정(512)되어 있기 때문에 길이가 긴 KorQuADv2의 데이터들을 학습하는데 어려움이 있다. 이를 해결하기 위해 **문서의 Context를 Segmentation하여 학습을 수행**하는 방식으로 모델을 설계하였다.

Pre-Training을 수행하기에는 실험 환경도 열악하고, monologg님께서 배포해주신 [KoELECTRA](https://github.com/monologg/KoELECTRA)의 성능이 매우 뛰어나기에 KoELECTRA를 Base로 모델을 설계하여 실험을 수행하였다. (monologg님께 감사드립니다!)

## Usage (Ubuntu)

### Dependencies

- bs4
- torch
- transformers

### Process Data

```bash
cd KorQuADv2/
for i in *.zip; do unzip $i;done

cd ../
python process_data.py
```

데이터 처리에는 **Context의 HTML Table Tag들을 처리**([Preprocessing 설명](https://github.com/decaf0cokes/KorQuADv2/blob/master/instruction/instruction.md))하고, **Context 및 Question들의 Encoding** 그리고 Train Set의 경우 **Answer Span의 Start/End Token Position을 추가**하는 과정을 포함한다.

위의 Code는 Tokenizer의 Encoding으로 인하여 수행 시간이 많이 소요된다. 하여 미리 처리한 데이터들을 .pkl 형태로 /pickles 디렉토리에 저장해 놓았다. 아래 Code는 git LFS를 통해 용량이 큰 파일을 내려받아 압축을 해제하는 동작을 한다. (위의 Code 수행 생략 가능) 아래 Code만으로 데이터 처리를 완료할 수 있다!

```bash
sudo apt-get install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs pull

cd pickles/
unzip contexts_encoded.pkl.zip
```

### Training Model

```bash
python run_training.py
```

### Prediction

Prediction Code는 임시로 작성되었음.

```bash
python run_prediction.py
```

KorQuADv2 Dev Set을 대상으로 예측을 수행하며, **EM 52.89/F1 66.48**의 성능을 보인다. Answer Span의 길이가 긴 경우 정답률이 극히 낮은 문제점을 노출한다.

## References

- [KorQuADv2](https://korquad.github.io/)
- [Huggingface](https://huggingface.co/transformers/)
- [KoELECTRA](https://github.com/monologg/KoELECTRA)
