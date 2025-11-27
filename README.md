# LogicKor

한국어 언어모델 다분야 사고력 벤치마크

## 변별력

최근 모델들의 성능이 높아진 탓에 상위권 모델에 대한 변별력이 없다는 문제가 있습니다. 해당 문제로 인해 원본 레포는 Read-Only로 전환되었으니 평가 시 참고하시기 바랍니다.

## Note

해당 레포는 거의 관리되지 않습니다.

## Repository

본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evaluation Example

GPU 0,1 사용, model_len 4096

### 1. 인퍼런스 결과 생성

```bash
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --gpu_devices 0,1 --model_len 4096
```

### 2. Judge 모델로 평가

#### OpenAI

```bash
python evaluator.py -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

#### Azure

```bash
export AZURE_ENDPOINT=$AZURE_ENDPOINT
export AZURE_DEPLOYMENT_NAME=$AZURE_DEPLOYMENT_NAME
export AZURE_API_VERSION=$AZURE_API_VERSION

python evaluator.py --azure -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

### 3. 결과 확인

```bash
python score.py -p ./evaluated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/default.jsonl
```
