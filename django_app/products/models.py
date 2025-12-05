# products/models.py
from django.db import models


class Product(models.Model):
    """
    상품 마스터 테이블

    - model_server 쪽 JSON / RDS의 product 테이블을 단순화해서 저장하는 용도
    - 즐겨찾기(FavoriteProduct)가 여기 FK를 잡고 있음
    """

    # JSON / RDS 에서 온 원본 상품 ID (예: guud_97008)
    external_id = models.CharField(
        "외부 상품 ID",
        max_length=64,
        null=True,
        blank=True,
        help_text="guud_97008 같은 크롤링 원본 상품 ID (없으면 비워둠)",
    )

    category = models.CharField("카테고리", max_length=64)
    brand_name = models.CharField("브랜드명", max_length=255, blank=True, default="")
    product_name = models.CharField("상품명", max_length=255)

    link_url = models.URLField("상품 링크 URL", max_length=1024)
    image_url = models.URLField(
        "대표 이미지 URL",
        max_length=1024,
        blank=True,
        default="",
    )

    description = models.TextField("상품 설명", blank=True, default="")

    price = models.PositiveIntegerField("가격(원)", default=0)

    # 무드 키워드 (예: ["아늑한", "우드톤", "북유럽"])
    mood_keywords = models.JSONField(
        "무드 키워드",
        default=list,
        blank=True,
        help_text="RAG / VLM 에서 추출한 무드 태그 리스트",
    )

    created_at = models.DateTimeField("등록일", auto_now_add=True)

    class Meta:
        ordering = ["id"]
        verbose_name = "상품"
        verbose_name_plural = "Products"

    def __str__(self) -> str:
        if self.brand_name:
            return f"{self.product_name} ({self.brand_name})"
        return self.product_name
