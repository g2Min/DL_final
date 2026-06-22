## 1. 프로젝트 설명

보통의 캐릭터 옷입히기 기능들은 미리 정해진 아이템들을 캐릭터에게 직접 탈부착하는 방식으로 이루어진다. 이런 경우 아이템이 너무 많아 바로 찾기가 어렵다는 단점이 있다. 이에 여러 foundation model 들을 사용하여 사용자가 직접참여하는 옷입히기 미니게임을 만들었다. <br/>

---

1. SAM으로 캐릭터  부위 마스킹
SAM 을 활용하여 6종의 동물 캐릭터 (다람쥐, 너구리, 곰, 고양이, 햄스터, 강아지) 의 상체/ 하체/ 전신 영역을 마스킹. <br/>
생성된 마스크는 `datasets/characters/masking/{character}` 에 저장되며 이후 diffusion 모델의 inpainting 단계에서 어느 신체 부위에 의상이 입힐지 사용됨.

2. Qwen2.5-7B + vLLM 서버로 빠른 LLM 추론
LLM 은 상황분석 (`SituationParser`)과 의상변환 (`GarmentAdapter`) 에 사용됨. <br/>
- 상황분석: 사용자가 입력한 자연어를 계절, 활동, 스타일, 색상 등의 구조화된 JSON으로 파싱함. <br/>
- 의상변환: 실제 패션 아이템을 "동물 캐릭터 의상" 으로 리사이징.

3. FashionCLIP 으로 의상 검색
미리 임베딩해둔 2000 개 이상의 의상 데이터를 바탕으로 사용자 조건에 알맞은 의상을 빠르게 검색함.

4. stable Diffusion 모델 + IP-adapter 로 의상 생성
stable-diffusion-xl-1.0-inpainting-0.1 모델을 사용하여 `3.` 에서 검색된 의상을 생성함. <br/>
IP-adapter 로 캐릭터 스타일에 맞게 좀더 정교하게 재현함.

> fastapi 를 활용하여 서버를 구축하고 react + vite 프레임워크로 클라이언트 UI를 구현함

## 2. 환경 세팅

> `vLLM` 서버 플랫폼을 구동하기 위해 `A-series` gpu 사양이 필요합니다.

아래 두 가지 방법 중 하나를 선택하여 환경을 구축할 수 있습니다.

```bash
git clone https://github.com/g2Min/DL_final.git
cd DL_final
```

#### 1) 로컬 환경 구축

```bash
conda create -n final python=3.11 -y
conda activate final
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.6 -c pytorch -c nvidia
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

#### 2) 컨테이너를 활용한 환경 구축

#### Build the Docker image

프로젝트 루트 디렉터리에서 실행합니다.

```bash
docker build -t final:1.0 .
```

#### Run the container

```bash
docker run -it --gpus all \
  --rm \
  --name dl_final \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  final:1.0
```

## 3. 실행 방법

### 1) 데이터셋 전처리

- characters 에 옷을 입힐 범위 masking (경로: datasets/characters/masking/{캐릭터 명})

  interactive SAM segmentation 모델을 사용하여 상의, 하의, 몸통 전체를 masking 한다.

- 캐릭터들에 입힐 의상 참조 데이터셋 다운로드

  hugging face 의 `ashraq/fashion-product-images-small` 를 다음 명령어로 받아온다.
  ```
  bash dataset.sh
  ```

- 의상 데이터셋 전처리

  캐릭터의 상의, 하의에 필요한 카테고리들만 따로 필터링
  ```
  # output: metadata.jsonl
  python scripts/prepare_garments.py
  ```

  필터링된 옷 datasets 들에 대해서 임베딩 진행
  ```
  # output: embeddings.npy
  python scripts/build_embeddings.py
  ```


### 2) vllm 서버 실행 및 모델 업로드
```
cd DL_fiinal
bash scripts/start_all.sh
```

### 3) 웹 클라이언트 실행
```
bash scripts/start_web.sh
```

### 4. 결과물

| 원본 | 프롬프팅 결과|
| --- | --- |
| ![배경화면](datasets/assets/report/original.png) | ![가을캠핑](datasets/assets/report/camping.png) |
