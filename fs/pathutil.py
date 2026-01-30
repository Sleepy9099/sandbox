# fs/pathutil.py
from __future__ import annotations

def normpath(p: str) -> str:
    if p == "":
        return "/"
    is_abs = p.startswith("/")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    out = []
    for x in parts:
        if x == "..":
            if out:
                out.pop()
        else:
            out.append(x)
    s = "/" + "/".join(out) if is_abs else "/".join(out)
    return s if s != "" else "/"

def abspath(cwd: str, p: str) -> str:
    if p.startswith("/"):
        return normpath(p)
    if cwd == "/":
        return normpath("/" + p)
    return normpath(cwd.rstrip("/") + "/" + p)
