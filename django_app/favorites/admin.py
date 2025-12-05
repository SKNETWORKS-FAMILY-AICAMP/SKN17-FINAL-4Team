# favorites/admin.py

from django.contrib import admin

from .models import FavoriteProduct, UserPreference


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "preferred_moods", "updated_at")
    search_fields = ("user__email",)


@admin.register(FavoriteProduct)
class FavoriteProductAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "product", "created_at")
    search_fields = ("user__email", "product__product_name")
    list_filter = ("created_at",)

