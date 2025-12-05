import pathlib
import re

ROOT = pathlib.Path("src")
pattern = re.compile(r'@\d+(?:\.\d+){1,3}(?=["\'])')

for path in ROOT.rglob("*.ts*"):
    text = path.read_text(encoding="utf-8")
    new_text = pattern.sub("", text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"Updated {path}")

