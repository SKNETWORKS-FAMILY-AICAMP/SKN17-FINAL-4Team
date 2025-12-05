# products/views.py
from rest_framework import generics, permissions
from rest_framework.filters import SearchFilter, OrderingFilter

from .models import Product
from .serializers import ProductSerializer


class ProductListView(generics.ListAPIView):
    """
    상품 목록 조회 (읽기 전용)

    - GET /api/products/
    - 쿼리 파라미터
      - category: 카테고리 필터
      - min_price, max_price: 가격 범위 필터
      - search: 상품명/브랜드명 검색
    """

    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [permissions.AllowAny]

    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ["product_name", "brand_name", "category"]
    ordering_fields = ["price", "created_at"]
    ordering = ["id"]

    def get_queryset(self):
        qs = super().get_queryset()

        category = self.request.query_params.get("category")
        min_price = self.request.query_params.get("min_price")
        max_price = self.request.query_params.get("max_price")

        if category:
            qs = qs.filter(category=category)

        if min_price is not None:
            try:
                qs = qs.filter(price__gte=int(min_price))
            except ValueError:
                pass

        if max_price is not None:
            try:
                qs = qs.filter(price__lte=int(max_price))
            except ValueError:
                pass

        return qs


class ProductDetailView(generics.RetrieveAPIView):
    """
    상품 상세 조회 (읽기 전용)

    - GET /api/products/<id>/
    """

    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [permissions.AllowAny]
