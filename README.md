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
- [11. 한 줄 회고](#한-줄-회고)


<br>
<br>

# 👥팀 소개

### 팀명: MOOD ON
> 당신의 무드를 이해해 오브제를 추천해주는 인테리어 감성 큐레이션 챗봇

### 팀원 소개
|[@김주영](https://github.com/samkim7788) | [@성기혁](https://github.com/venus241004) | [@양정민](https://github.com/Yangmin3) | [@이가은](https://github.com/Leegaeune) | [@임산별](https://github.com/ImMountainStar) |[@주수빈](https://github.com/Subin-Ju)|
|----------------------|----------------------|----------------------|----------------------|-----------------------|----------------------|
| <img src="readme_image/profile/김주영.jpg" width="150" height="150" /> | <img src="readme_image/profile/성기혁.jpg" width="150" height="150"> | <img src="readme_image/profile/양정민.jpg" width="150" height="150"> | <사진> | <사진> | <img src="readme_image/profile/주수빈.jpg" width="150" height="150"> |

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

### 시장 분석
![인테리어 시장 구분](readme_image/인테리어%20시장.png)

![홈퍼니싱 시장](readme_image/홈퍼니싱%20시장.png)
> 출처: 보스턴컨설팅그룹  


- 홈퍼니싱 시장
    - 정의: 개인의 취향과 감성에 맞게 공간을 꾸미는 인테리어 시장 
    - 특징: 낮은 진입장벽 & 높은 일반 소비자의 참여도  

→ 소비가 가장 활발히 이루어지고, 추천의 전환 효과가 큰 시장  
→ 2018년 이후 꾸준히 성장 중  
→ 기존 이커머스 시장(ex: 오늘의집, 무신사, 에이블리 등)들의 인테리어 카테고리 확장

<br>

### 소비자 분석
![소비자 분석](readme_image/journey%20map.png)
**pain point**
1. 초기 탐색 단계부터 상품이 너무 많아서 어떤 무드인지 판단 불가
2. 비교 탐색 단계에서는 원하는 분위기의 상품을 직접 찾아야 하는 번거로움
3. 구매 단계에서는 내 방에 어울릴지 확신할 수 없음
4. 구매 후에는 이미지와 실제 분위기 차이로 인한 실망  

→ **본인의 무드·취향을 스스로 정의하기 어려운 문제**에서 비롯되는 현상
<br>

### 경쟁사 분석


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

<br>
<br>

# 기능 및 화면 설계

<br>
<br>

# AI/추천 시스템 설계

<br>
<br>

# 프로젝트 개선 노력

<br>
<br>

# 수행 결과 및 시연 영상

<br>
<br>

# 한 줄 회고
- 김주영: 
- 성기혁:
- 양정민: 
- 이가은: 
- 임산별: 
- 주수빈: 