# favorites/views.py

from django.shortcuts import get_object_or_404
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import FavoriteProduct, UserPreference
from .serializers import FavoriteProductSerializer, UserPreferenceSerializer


class UserPreferenceView(APIView):
    """
    사용자 선호도 설문 조회/저장/수정

    - GET  /api/favorites/preferences/     → 현재 사용자의 설문 결과 조회
    - POST /api/favorites/preferences/     → 최초 설문 저장
    - PUT  /api/favorites/preferences/     → 전체 수정 (재설문)
    - PATCH /api/favorites/preferences/    → 부분 수정
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        pref, created = UserPreference.objects.get_or_create(user=request.user)
        serializer = UserPreferenceSerializer(pref)
        return Response(serializer.data)

    def post(self, request):
        # 이미 있으면 덮어쓰기 (재설문과 동일)
        pref, created = UserPreference.objects.get_or_create(user=request.user)
        serializer = UserPreferenceSerializer(
            pref,
            data=request.data,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request):
        pref, created = UserPreference.objects.get_or_create(user=request.user)
        serializer = UserPreferenceSerializer(
            pref,
            data=request.data,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)

    def patch(self, request):
        pref, created = UserPreference.objects.get_or_create(user=request.user)
        serializer = UserPreferenceSerializer(
            pref,
            data=request.data,
            partial=True,
        )
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)


class FavoriteProductListCreateView(APIView):
    """
    관심 상품 목록 조회 및 등록.

    - GET  /api/favorites/       → 내 관심 상품 목록 조회
    - POST /api/favorites/       → 관심 상품 등록 (body: { "product_id": 123 })
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        favorites = FavoriteProduct.objects.filter(user=request.user).select_related("product")
        serializer = FavoriteProductSerializer(favorites, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = FavoriteProductSerializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)
        favorite = serializer.save()
        out = FavoriteProductSerializer(favorite).data
        return Response(out, status=status.HTTP_201_CREATED)


class FavoriteProductDeleteView(APIView):
    """
    관심 상품 삭제.

    - DELETE /api/favorites/<id>/    → 해당 관심 상품 삭제
    """

    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, pk: int):
        favorite = get_object_or_404(FavoriteProduct, pk=pk, user=request.user)
        favorite.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
