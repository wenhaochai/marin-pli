import json, sys, urllib.request, time

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def load_all(domain, page_id):
    blocks = {}
    cursor = {"stack": []}
    for chunk in range(200):
        d = post(f"https://{domain}/api/v3/loadCachedPageChunkV2" if False else f"https://{domain}/api/v3/loadPageChunk",
                 {"pageId": page_id, "limit": 100, "cursor": cursor, "chunkNumber": chunk, "verticalColumns": False})
        rm = d.get("recordMap", {})
        for k, v in rm.get("block", {}).items():
            blocks[k] = v
        cursor = d.get("cursor", {"stack": []})
        if not cursor.get("stack"):
            break
    return blocks

def text_of(prop):
    out = []
    for seg in prop or []:
        t = seg[0]
        anns = seg[1] if len(seg) > 1 else []
        for a in anns:
            if a[0] == 'e':  # inline equation
                t = f"${a[1]}$"
            if a[0] == 'a':
                t = f"{t} <{a[1]}>"
        out.append(t)
    return ''.join(out)

def render(blocks, bid, depth=0, seen=None, out=None):
    seen = seen if seen is not None else set(); out = out if out is not None else []
    if bid in seen or bid not in blocks: return out
    seen.add(bid)
    b = blocks[bid].get("value", {})
    if "value" in b and isinstance(b["value"], dict): b = b["value"]
    t = b.get("type"); p = b.get("properties", {})
    title = text_of(p.get("title"))
    ind = "  " * depth
    if t in ("header", "sub_header", "sub_sub_header"):
        out.append("\n" + {"header": "# ", "sub_header": "## ", "sub_sub_header": "### "}[t] + title)
    elif t == "equation":
        out.append(ind + "$$" + title + "$$")
    elif t == "image":
        src = (p.get("source") or [[""]])[0][0]
        cap = text_of(p.get("caption"))
        out.append(ind + f"[IMG {src[-60:]} | {cap}]")
    elif t == "table_row":
        cells = [text_of(v) for k, v in p.items()]
        out.append(ind + " | ".join(cells))
    elif t == "code":
        out.append(ind + "```\n" + title + "\n```")
    elif t == "page" and depth > 0:
        out.append(ind + f"[SUBPAGE {title}]")
        return out
    else:
        if title: out.append(ind + ("- " if t in ("bulleted_list", "numbered_list") else "") + title)
        elif t not in ("page", "column_list", "column", "table", "divider", None): out.append(ind + f"[{t}]")
    for c in b.get("content", []) or []:
        render(blocks, c, depth + (1 if t not in ("page", "column_list", "column") else 0), seen, out)
    return out

domain, page_id, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
blocks = load_all(domain, page_id)
json.dump(blocks, open(outpath + ".json", "w"))
lines = render(blocks, page_id)
open(outpath, "w").write("\n".join(lines))
print(len(blocks), "blocks;", len(lines), "lines ->", outpath)
