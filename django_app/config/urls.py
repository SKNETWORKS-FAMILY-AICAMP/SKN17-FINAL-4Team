# config/urls.py

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),

    # API 엔드포인트
    path("api/accounts/", include("accounts.urls")),
    path("api/chat/", include("chat.urls")),
    path("api/products/", include("products.urls")),
    path("api/favorites/", include("favorites.urls")),
]

# 개발 환경에서 media 파일 서빙
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
