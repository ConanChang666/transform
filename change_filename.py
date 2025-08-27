import os, re
from pathlib import Path

# 目前檔案格式: "YYYYMMDD stockid.txt"
PATTERN = re.compile(r'^(?P<date>\d{8})\s+(?P<stock>\d{4,6}|[A-Za-z\.]{1,10})(\.txt)$')

folder = "/Users/fiiconan/Desktop/transform/file_ready_to_transform"   # 你的檔案目錄
for f in Path(folder).glob("*.txt"):
    m = PATTERN.match(f.name)
    if not m:
        continue
    date = m.group("date")
    stock = m.group("stock")
    new_name = f"{stock}_{date}.txt"   # 改成 stockid_YYYYMMDD.txt
    f.rename(f.with_name(new_name))
    print(f"Renamed {f.name} -> {new_name}")