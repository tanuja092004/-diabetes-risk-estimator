"""Inline model.json into index.template.html -> index.html (a single static file)."""
import json
model = json.load(open("model.json"))
html = open("index.template.html", encoding="utf-8").read()
html = html.replace("/*__MODEL__*/null", json.dumps(model, separators=(",", ":")))
open("index.html", "w", encoding="utf-8").write(html)
print("index.html written:", len(html) // 1024, "KB")
