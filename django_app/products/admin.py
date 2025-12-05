# products/admin.py
from django.contrib import admin

from .models import Product


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "product_name",
        "brand_name",
        "category",
        "price",
        "created_at",
    )
    list_filter = ("category", "brand_name")
    search_fields = ("product_name", "brand_name", "category", "external_id")
    ordering = ("id",)
