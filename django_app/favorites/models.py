# favorites/models.py
from django.conf import settings
from django.db import models


class UserPreference(models.Model):
    """
    사용자 선호도 설문 결과.

    - user: 1:1
    - preferred_moods: 사용자가 선택한 무드 라벨 (최소 1개, 최대 3개) 리스트
    - created_at / updated_at: 설문 시각
    """

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="preference",
    )
    preferred_moods = models.JSONField(
        default=list,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "사용자 선호도"
        verbose_name_plural = "사용자 선호도"

    def __str__(self) -> str:
        return f"Preference({self.user.email})"


class FavoriteProduct(models.Model):
    """
    관심 상품 (찜).

    - user: 찜한 사용자
    - product: products.Product 와의 FK
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="favorite_products",
    )
    # products 앱의 Product 모델 사용 (app_label.ModelName 문자열로 지정)
    product = models.ForeignKey(
        "products.Product",
        on_delete=models.CASCADE,
        related_name="favorited_by",
        null=True,      # ★ 여기 추가
        blank=True,     # ★ 여기 추가
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "관심 상품"
        verbose_name_plural = "관심 상품"
        constraints = [
            models.UniqueConstraint(
                fields=["user", "product"],
                name="unique_favorite_per_user",
            )
        ]

    def __str__(self) -> str:
        return f"{self.user.email} ❤️ {self.product_id}"
