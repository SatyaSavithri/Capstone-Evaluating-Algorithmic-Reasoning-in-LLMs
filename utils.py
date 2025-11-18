# utils.py
import json, re

def parse_final_json_path(text: str):
    """
    Attempt to extract the JSON object with key "path" from text.
    Fallback: find last line containing 'Room' tokens and parse arrow/comma separated.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        if "{" in line and "path" in line:
            try:
                i = line.find("{"); j = line.rfind("}")
                js = line[i:j+1] if (i>=0 and j>i) else line
                parsed = json.loads(js)
                p = parsed.get("path", [])
                if isinstance(p, list):
                    return p
            except Exception:
                continue
    for line in reversed(lines):
        if "Room" in line:
            parts = re.split(r'->|,|;', line)
            parts = [p.strip() for p in parts if "Room" in p]
            if parts:
                return parts
    return []
