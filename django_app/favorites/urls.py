# favorites/urls.py

from django.urls import path

from . import views

urlpatterns = [
    # 사용자 선호도 설문
    path("preferences/", views.UserPreferenceView.as_view(), name="user-preferences"),
    # 관심 상품 목록/등록
    path("", views.FavoriteProductListCreateView.as_view(), name="favorite-list-create"),
    # 관심 상품 삭제
    path("<int:pk>/", views.FavoriteProductDeleteView.as_view(), name="favorite-delete"),
]

