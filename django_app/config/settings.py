# config/settings.py

import os
from pathlib import Path

# 프로젝트 기본 경로 (django_app 폴더)
BASE_DIR = Path(__file__).resolve().parent.parent

# 개발용 시크릿 키 (실서비스에서는 환경변수로 빼기!)
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-this-in-production",
)

DEBUG = True

# 개발 단계에서는 전체 허용
ALLOWED_HOSTS: list[str] = ["*"]

# 애플리케이션 설정
INSTALLED_APPS = [
    # Django 기본 앱
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # 서드파티 앱
    "rest_framework",
    "corsheaders",

    # 로컬 앱
    "accounts",
    "chat",
    "favorites",
    "products",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",  # CORS가 CommonMiddleware보다 먼저
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # 필요하면 템플릿 디렉터리 사용
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ======================
# Database
# ======================

# 개발용: sqlite3
# 추후 RDS(MySQL)로 바꾸려면 이 블록을 교체하면 됨.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# 예시: 추후 RDS(MySQL)로 전환 시
# DATABASES = {
#     "default": {
#         "ENGINE": "django.db.backends.mysql",
#         "NAME": os.environ.get("DB_NAME", "moodon"),
#         "USER": os.environ.get("DB_USER", "moodon_app"),
#         "PASSWORD": os.environ.get("DB_PASSWORD", ""),
#         "HOST": os.environ.get("DB_HOST", "localhost"),
#         "PORT": os.environ.get("DB_PORT", "3306"),
#         "OPTIONS": {
#             "charset": "utf8mb4",
#         },
#     }
# }

# ======================
# Auth
# ======================

AUTH_USER_MODEL = "accounts.User"

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 6},
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ======================
# 국제화
# ======================

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

# ======================
# Static & Media
# ======================

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [
    BASE_DIR / "static",
]

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ======================
# Django REST Framework
# ======================

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
}

# ======================
# CORS / CSRF
# ======================

# 프론트엔드가 다른 포트(5173/5174 등)에서 구동되기 때문에
# withCredentials + SessionAuthentication 조합이 동작하도록
# 명시적으로 허용할 도메인과 자격 증명 사용을 설정한다.
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

# ======================
# 이메일 설정 (개발용: 콘솔 출력)
# ======================

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DEFAULT_FROM_EMAIL = "no-reply@moodon.local"

# ======================
# 모델 서버 URL
# ======================

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8001")

# ======================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

