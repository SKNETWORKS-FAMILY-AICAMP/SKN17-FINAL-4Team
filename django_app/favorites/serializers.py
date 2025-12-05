# favorites/serializers.py

from rest_framework import serializers

from .models import FavoriteProduct, UserPreference
from products.models import Product


# ---------- 선호도 설문 ----------


class UserPreferenceSerializer(serializers.ModelSerializer):
    """
    사용자 선호도 설문용 시리얼라이저.

    - preferred_moods: 문자열 리스트 (예: ["모던", "우드톤", "내추럴"])
    """

    preferred_moods = serializers.ListField(
        child=serializers.CharField(max_length=50),
        allow_empty=False,
    )

    class Meta:
        model = UserPreference
        fields = ["preferred_moods", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

    def validate_preferred_moods(self, value):
        if len(value) > 3:
            raise serializers.ValidationError("선호 무드는 최대 3개까지 선택할 수 있습니다.")
        # 중복 제거
        unique = list(dict.fromkeys(value))
        return unique


# ---------- 관심 상품 ----------


class ProductSimpleSerializer(serializers.ModelSerializer):
    """관심 상품 목록에 사용할 간단한 Product 정보."""

    class Meta:
        model = Product
        fields = ["id", "brand_name", "product_name", "image_url", "link_url", "price"]


class FavoriteProductSerializer(serializers.ModelSerializer):
    product = ProductSimpleSerializer(read_only=True)
    product_id = serializers.PrimaryKeyRelatedField(
        queryset=Product.objects.all(),
        source="product",
        write_only=True,
    )

    class Meta:
        model = FavoriteProduct
        fields = ["id", "product", "product_id", "created_at"]
        read_only_fields = ["id", "product", "created_at"]

    def create(self, validated_data):
        user = self.context["request"].user
        product = validated_data["product"]
        favorite, _ = FavoriteProduct.objects.get_or_create(
            user=user,
            product=product,
        )
        return favorite

