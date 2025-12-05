# products/serializers.py
from rest_framework import serializers

from .models import Product


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = [
            "id",
            "external_id",
            "category",
            "brand_name",
            "product_name",
            "link_url",
            "image_url",
            "description",
            "price",
            "mood_keywords",
            "created_at",
        ]
