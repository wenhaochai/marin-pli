"""Glossary check of the polished page: counts of competing terms in the Chinese and English units (units.json from
extract_units.py run on the polished page). Usage: term_check.py <units.json>"""
import json, re, sys
u = json.load(open(sys.argv[1]))["units"]
zh = "".join(x["shown"] for x in u if x["lang"] == "zh"); en = " ".join(x["shown"] for x in u if x["lang"] == "en")
pairs_zh = [("骨干", "主干"), ("承受全部", "承担全部"), ("源", "来源"), ("query", "查询"), ("运行", " run"), ("下一个词", "下一个 token"),
            ("下下个词", "下下个 token"), ("embedding", "嵌入"), ("独立头", "独立的头"), ("普通模型", "常规模型"), ("探针", "探头"),
            ("结构", "架构"), ("读出", "读取"), ("付", "代价")]
for a, b in pairs_zh: print(f"ZH {a!r}: {zh.count(a)}   {b!r}: {zh.count(b)}")
for t in ["付 0", "付出", "付得", "付全部", "付的"]: print(f"ZH {t!r}: {zh.count(t)}")
for a, b in [("output head", "final head"), ("separate-heads model", "separate head model"), ("ordinary model", "plain model"),
             ("Hyper-Connections", "HC "), ("the start", "initial value"), ("fa4thd", "dlogits")]:
    print(f"EN {a!r}: {len(re.findall(re.escape(a), en))}   {b!r}: {len(re.findall(re.escape(b), en))}")
print("EN em dash:", en.count("—"), " ZH em dash:", zh.count("——"), zh.count("—"))
