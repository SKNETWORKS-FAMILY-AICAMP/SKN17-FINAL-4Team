import json

with open("products_all_ver1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

unique_categories = sorted(set(d["category_id"] for d in data))
print(unique_categories)
