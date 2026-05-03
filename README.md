# SKN17 Final 4Team - MOOD ON

- **대주제**: LLM 활용 대화형 상품 추천
- **프로젝트 기간**: 25.10.28 ~ 25.12.18

<br>

# 🗂️ 목차

- [01. 팀 소개](#👥팀-소개)
- [02. 프로젝트 개요](#프로젝트-개요)
- [03. 기술 스택](#기술-스택)
- [04. WBS](#wbs)
- [05. 시스템 아키텍처](#시스템-아키텍처)
- [06. 데이터 설계](#데이터-설계)
- [07. 기능 및 화면 설계](#기능-및-화면-설계)
- [08. AI/추천 시스템 설계](#ai추천-시스템-설계)
- [09. 프로젝트 개선 노력](#프로젝트-개선-노력)
- [10. 수행 결과 및 시연 영상](#수행-결과-및-시연-영상)


<br>
<br>

# 👥팀 소개

### 팀명: MOOD ON
> 당신의 무드를 이해해 오브제를 추천해주는 인테리어 감성 큐레이션 챗봇

### 팀원 소개
|[@김주영](https://github.com/samkim7788) | [@성기혁](https://github.com/venus241004) | [@양정민](https://github.com/Yangmin3) | [@이가은](https://github.com/Leegaeune) | [@임산별](https://github.com/ImMountainStar) |[@주수빈](https://github.com/Subin-Ju)|
|----------------------|----------------------|----------------------|----------------------|-----------------------|----------------------|


<br>
<br>

# 프로젝트 개요

## 프로젝트 명
- 🪴 **MOODON** - 당신의 무드를 이해해 오브제를 추천해주는 인테리어 감성 큐레이션 챗봇

## 프로젝트 배경
소비자들은 인테리어 제품을 구매할 때 단순히 가격과 브랜드와 같은 정략적인 요소만으로 선택하지 않습니다.  
인테리어는 그 공간을 대표하는 상품으로서 **감성**과 **분위기**같은 정성적인 요소가 중요한 요소로서 작용합니다.  
이러한 특성 때문에 소비자들은 자신이 원하는 무드와 어울리는 상품을 찾는 데 어려움을 겪곤 합니다. 
따라서 저희 팀은 인테리어 시장의 구조를 분석하고, 실제 소비자들이 겪는 불편함을 파악하여, 이 문제를 해결할 수 있는 인테리어  **상품 추천 챗봇**을 기획하고자 했습니다.   

### 타겟 시장
| 인테리어 시장 구분 | 홈퍼니싱 시장 |
|---|---|
| ![인테리어 시장 구분](readme_image/인테리어%20시장.png) | ![홈퍼니싱 시장](readme_image/홈퍼니싱%20시장.png) |

> 출처: 보스턴컨설팅그룹

홈퍼니싱 시장은 2018년 이후 꾸준히 성장 중이며,  
특히 코로나19를 기점으로 다양한 사회적 변화와 ‘집 꾸미기’ 트렌드의 확산에 영향을 받아  
기존 이커머스들(예: 오늘의집, 무신사·에이블리 등)까지 인테리어 카테고리를 확장 

<br>

### 소비자 분석
| Pain Point | Solution |
|---|---|
| **Pain Point** <br><br> ![소비자 분석-문제](readme_image/소비자%20분석(문제).png) <br><br> 1. 초기 탐색 단계: 상품 과다로 무드 판단 어려움 <br> 2. 비교 탐색 단계: 원하는 분위기 상품 직접 탐색 부담 <br> 3. 구매 단계: 공간 및 취향 일치 여부 확신 부족 <br> 4. 구매 후: 이미지와 실제 분위기 차이로 인한 불만 | **Solution** <br><br> ![소비자 분석-해결](readme_image/소비자분석(솔루션).png) <br><br> 1. 이미지 기반 무드 자동 추출 <br> 2. 채팅 및 선호도 기반 맞춤 상품 추천 <br> 3. 사용자 공간 이미지 기반 추천 제공 <br> 4. 구매 전 분위기 예측 기반 만족도 개선 |



<br>

## 🚨 Problem - Solution

**Problem**
- 감성 맥락 부재 → 기존 플랫폼은 가격/브랜드 중심 필터만 제공  
- 정보 과부하 → 사용자가 직접 상품을 비교·탐색해야 하는 부담  
- 맞춤형 추천 한계 → 구매이력 기반 추천으로 실제 취향 반영 부족  

**Solution**
- 사용자 공간 이미지 기반 감성·무드 추천 제공  
- 챗봇 기반 탐색 → 선택 → 구매링크까지 전 과정 지원  
- 채팅 히스토리 + 선호도 학습 기반 개인화 상품 추천


<br>

## 프로젝트 목표
### 프로젝트 목표

 1️⃣ LLM을 활용해 사용자에게 SEAMLESS 추천 경험을 제공하고, 피드백을 받아 성능 최적화 <br>
 2️⃣ 챗봇 내 이미지 업로드 기능을 통해 사용자의 방 이미지를 분석하여 맞춤 감성 추천  <br>
 3️⃣ 레퍼런스로 활용할 수 있는 무드 이미지를 제공해 사용자 맞춤 인테리어 아이디어를 제공  <br>
 



<br>
<br>

# 기술 스택
| 카테고리 | 기술 스택 |
|----------|-------------------------------------------|
| **언어** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) |
| **프레임워크** | ![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white) |
| **벡터 데이터베이스** | ![CHROMA](https://img.shields.io/badge/FAISS-009688?style=for-the-badge&logo=Apache&logoColor=white) |
| **LLM 모델/프레임워크** | ![Qwen2.5-14B-KOREAN](https://img.shields.io/badge/Qwen2.5--14B--KOREAN-FFB000?style=for-the-badge&logo=OpenAI&logoColor=white) ![LANGCHAIN](https://img.shields.io/badge/LangChain-005F73?style=for-the-badge&logo=Chainlink&logoColor=white) ![LANGGRAPH](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=Chainlink&logoColor=white) |
| **VLM 모델** | ![QWEN3-VL-8B-Instruct](https://img.shields.io/badge/QWEN3_VL_8B_Instruct-6C63FF?style=for-the-badge&logo=Apache&logoColor=white)
| **임베딩 모델** | ![TEXT-EMBEDDING-3-LARGE](https://img.shields.io/badge/TEXT--EMBEDDING--3--LARGE-8C9E90?style=for-the-badge&logo=OpenAI&logoColor=white) |
| **데이터베이스** | ![AWS RDS](https://img.shields.io/badge/AWS_RDS-5B6CFF?style=for-the-badge&logo=amazonaws&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=white) |
| **스토리지** | ![AMAZON S3](https://img.shields.io/badge/AMAZON_S3-000000?style=for-the-badge&logo=amazonaws&logoColor=white) |
| **통합 개발 환경** | ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |
| **UI 및 프론트엔드** | <img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> <img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"> <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"> |
| **배포 및 컨테이너** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white) ![Docker Compose](https://img.shields.io/badge/Docker--Compose-1488C6?style=for-the-badge&logo=Docker&logoColor=white) |
| **실행 환경** | ![RunPod](https://img.shields.io/badge/RunPod-FF4500?style=for-the-badge&logo=Render&logoColor=white) ![AWS EC2](https://img.shields.io/badge/AWS%20EC2-FF9900?style=for-the-badge&logo=Amazon%20AWS&logoColor=white) |
| **협업 및 형상관리** |  ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white) ![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white) ![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=Google%20Drive&logoColor=white) |

<br>
<br>

# WBS
![WBS](readme_image/1212WBS.png)  

<br>
<br>

# 시스템 아키텍처
![시스템 아키텍처](readme_image/시스템%20아키텍쳐%20최종본.png)

<br>
<br>

# 데이터 설계
<img width="1230" height="1212" alt="image" src="https://github.com/user-attachments/assets/d5eb1c08-5bee-4d1d-8be4-bcff24ecf486" />

<br>
<br>

# 기능 및 화면 설계
#### 서비스 기능 Flow 
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/44f6e57b-3620-4108-9374-6b3f9132df56" />


#### 화면설계 
<img width="677" height="365" alt="image" src="https://github.com/user-attachments/assets/8ec01bae-21c9-495c-af0a-65b3c91a955a" />

<br>
<br>

# AI/추천 시스템 설계
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/6b9848dd-1df7-4ae8-bb6a-9f849dea32cd" />

## 모델 선정

### 모델 선정 기준

* 한국어 질의 응답 정확도
* RAG 결합 시 응답 안정성
* Hallucination 최소화 수준
* 이미지 기반 무드 및 소재 인식 성능
* 추천 서비스 적용 가능성

---

### 최종 선정 모델

| 구분  | 모델                     |
| --- | ---------------------- |
| LLM | Qwen2.5-14B-Korean     |
| VLM | Qwen2.5-VL-7B-Instruct |

---

### LLM 선정 근거

* 한국어 문맥 이해 성능 우수
* RAG 환경에서 응답 안정성 확보
* Hallucination 발생 최소 수준
* 응답 속도 안정성 확보

---

### VLM 선정 근거

* 소재 / 질감 / 색감 인식 성능 우수
* 이미지 기반 무드 해석 가능
* 텍스트 설명 생성 품질 안정성
* 추천 서비스 적용 적합성

---

## 모델 성능 평가

### 평가 방식

* 사용자 텍스트 + 이미지 입력 기반 평가
* RAG 검색 + 이미지 무드 추출 결합 평가
* 실제 추천 시나리오 기반 테스트

---

### 성능 평가 결과

| 평가 항목         | 결과                     |
| ------------- | ---------------------- |
| 텍스트 검색 유사도    | 평균 cosine ≈ 0.59       |
| RAG 응답 안정성    | PASS                   |
| 이미지-텍스트 맥락 연결 | 우수                     |
| 무드 / 스타일 반영   | 우수                     |
| 평균 응답 시간      | 35~45s (g6e.xlarge 기준) |

---

### 최종 선정 결과

* 한국어 기반 추천 응답 안정성 확보
* 이미지 기반 무드 해석 성능 확보
* 멀티모달 추천 서비스 적용 가능성 검증



<br>
<br>


# 프로젝트 개선 노력

<br>
<br>

# 수행 결과 및 시연 영상
## 평가 
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/2a90978c-e9fd-4f31-bab0-27405c0d0249" />

## 🎬 시연 영상

👉 [시연영상 보러가기](https://drive.google.com/file/d/1-qSSbeupVFnMLRqk-OgZ-P_xRg5Hvw2e/view?usp=drive_link)


<br>
<br>

---
# 기대효과 및 BM
![기대효과](readme_image/기대효과.png)
**기대효과**
- 사용자 편의성 증대
- 홈퍼니싱 시장 내 경쟁 우위

**Business Model**
- 입점 수수료(CPC)
- 브랜드 광고 수수료(CPM)
---

# 향후 계획

- 멀티모달 추천 정확도 고도화  
- 사용자 취향 기반 개인화 추천 모델 적용  
- 인테리어 / 가구 데이터셋 확장  
- 추천 Ranking 및 필터링 로직 개선  
- 실시간 추천 응답 속도 최적화  
- 사용자 행동 데이터 기반 추천 반영  
- Vector DB 검색 성능 최적화  
- 서비스 배포 환경 안정화  

---
