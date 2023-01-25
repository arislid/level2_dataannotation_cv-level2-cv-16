# **Data Annotation & OCR Detection Project**

![Main](https://user-images.githubusercontent.com/103131249/214512646-bd6acd0d-17e6-4884-9204-cce8585bcb71.png)

## 📰 **Contributors**

**CV-16조 💡 비전길잡이 💡**</br>NAVER Connect Foundation boostcamp AI Tech 4th

|민기|박민지|유영준|장지훈|최동혁|
|:----:|:----:|:----:|:---:|:---:|
|[<img alt="revanZX" src="https://avatars.githubusercontent.com/u/25689849?v=4&s=100" width="100">](https://github.com/revanZX)|[<img alt="arislid" src="https://avatars.githubusercontent.com/u/46767966?v=4&s=100" width="100">](https://github.com/arislid)|[<img alt="youngjun04" src="https://avatars.githubusercontent.com/u/113173095?v=4&s=100" width="100">](https://github.com/youngjun04)|[<img alt="FIN443" src="https://avatars.githubusercontent.com/u/70796031?v=4&s=100" width="100">](https://github.com/FIN443)|[<img alt="choipp" src="https://avatars.githubusercontent.com/u/103131249?v=4&s=117" width="100">](https://github.com/choipp)|
|AI hub</br>SynthTextKR | ICDAR</br>SynthText | Augmentation</br>SynthText | ICDAR</br>Optimization | SynthText 500k</br>ICDAR 17/19|
</br>


## 📰 **Links**

- [비전 길잡이 Notion 📝](https://vision-pathfinder.notion.site/b90e838e2bc24dccb97e7e7e578c0191)
- [비전 길잡이 발표자료 & WrapUpReport](./appendix/)

## 📰 **Result**

![Result](https://user-images.githubusercontent.com/103131249/214503241-f2105573-aaae-4c7f-a2a7-d5795ec883ba.png)

---

## 📰 **Model**

![image](https://user-images.githubusercontent.com/103131249/214510914-90e32259-e766-4537-9ab3-df3213d4ae36.png)

- [EAST: An Efficient and Accurate Scene Text Detector](https://github.com/SakuraRiven/EAST)
- Data-driven approach를 통한 글자 검출 성능 향상 목표 - 모델 고정

## 📰 **Strategy**

![Strategy](https://user-images.githubusercontent.com/103131249/214511736-cebe9c5a-83b5-4f4a-898a-38f5da3b2129.png)

- **ImageNet pretrained Backbone** + **대량의 합성 데이터 pre-training** + **fine-tuning**
- SynthText pre-generated 데이터셋 확보 후 pre-training
- 이후 ICDAR 17/19 데이터로 fine-tuning하여 좋은 성능 확인

## 📘 **Dataset**

![image](https://user-images.githubusercontent.com/113173095/214503526-04a7e69e-fa9c-4bad-b0c0-293bae4475c4.png)

- boostcamp 자체 annotation 데이터셋 포함 4개 범주 데이터셋 활용

### **📘 (1) AI Hub**

![Aihub_sample](https://user-images.githubusercontent.com/46767966/214509698-c4c36a63-7df1-4072-8875-abb33b9d747d.png)

- **공공행정문서 OCR** : 카테고리 별 8장 약 2618장의 데이터셋 활용
- **야외 실제 촬영 한글 이미지**  : EDA 결과 적합하지 않아 제외

### **📘 (2) ICDAR**

![ICDAR](https://user-images.githubusercontent.com/103131249/214517208-7b4583a6-a678-4673-a933-0a9beaa2506b.png)

- **ICDAR17 MLT** - 9개 언어, Training 7,200장, Validation 1,800장
- **ICDAR19 MLT** - 10개 언어, Training 10,000장

### **📘 (3) SynthText**

![SynthText](https://user-images.githubusercontent.com/103131249/214511477-0b25d967-8cf3-46c3-b8d6-f9b2420b9c7e.png)

- **Synthetic data**: 영어 858,750장 중 542,706장 학습에 사용
- **E2E-MLT Data**: 한국어 40,432장 중 5,452장을 학습 데이터 사용
- 규모가 큰 데이터셋(영어) pre-trained 후 fine-tuning하는 전략
- 이후 5,452장(한국어) + 7,939장(영어) 소규모 competition 데이터셋 pre-trained 적용

## 📰 **Experiments**

### **1. AiHub OCR**

| exp                       | recall</br>(public) | precision</br>(public) | f1-score</br>(public) | f1-score</br>(private) |
| ------------------------- | -------------------- | ----------------------- | ---------------------- | ----------------------- |
| AI_hubOCR                 | 0.5958               | 0.3876                  | 0.4690                 | 0.5097                  |
| pre-AI_hubOCR + </br>ICDAR1719 | 0.4890               | 0.7001                  | 0.5758                 | 0.6087                  |

### **2. ICDAR Only**

| exp           | recall</br>(public) | precision</br>(public) | f1-score</br>(public) | f1-score</br>(private) |
| ------------- | -------------------- | ----------------------- | ---------------------- | ----------------------- |
| ICDAR 17      | 0.5510               | 0.7877                  | 0.6415                 | 0.6309                  |
| ICDAR 19      | 0.5739               | 0.8141                  | 0.6739                 | 0.6730                  |
| ICDAR 19 Norm | 0.5931               | 0.7899                  | 0.6775                 | 0.6793                  |
| ICDAR 17, 19  | 0.5851               | 0.8095                  | 0.6792                 | 0.6688                  |

### **3. Pre-trained SynthText data**

| exp                               | recall</br>(public) | precision</br>(public) | f1-score</br>(public) | f1-score</br>(private) |
| --------------------------------- | -------------------- | ----------------------- | ---------------------- | ----------------------- |
| ICDAR 19, ST-kr                   | 0.5379               | 0.7793                  | 0.6365                 | 0.6794                  |
| pre-en(500k) + ICDAR 19           | 0.5749               | 0.8141                  | 0.6739                 | 0.6730                  |
| pre-en(500k) + ICDAR 17/19        | 0.5997               | 0.8080                  | 0.6884                 | 0.7192                  |
| pre-en(500k) + ICDAR 17/19, ST-kr | 0.5815               | 0.7638                  | 0.6603                 | 0.6707                  |
| pre-mix + ICDAR 17/19             | 0.6113               | 0.8171                  | 0.6993                 | 0.7115                  |
| pre-en(80k) + ICDAR 17/19         | 0.5583               | 0.7458                  | 0.6386                 | 0.6507                  |

## 📰 **LB Timeline ⌛**

![LB Timeline](https://user-images.githubusercontent.com/103131249/214514024-e8c98ae4-c446-4fa9-a343-e530369c6964.png)

- 초반에 ICDAR 17, 19 적용하여 높은 점수 확보
- SynthText 적용 후 ImageNet pretrained Backbone + 대량의 합성 데이터 pretrain
- 최종적으로 fine-tuning 통해 후반부에 성능 끌어올림

## 📰 **Directory Structure**

```
|-- 🗂 appendix : 발표자료 및 WrapUpReport
|-- 🗂 code     : 학습시 사용했던 코드
`-- README.md
```