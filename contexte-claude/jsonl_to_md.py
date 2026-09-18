#!/usr/bin/env python3
"""Convertit les transcripts Claude Code (.jsonl) en Markdown lisible.

    python3 contexte-claude/jsonl_to_md.py

Lit ~/.claude/projects/-home-anass-GenerativePDE/*.jsonl et écrit un .md par
conversation dans contexte-claude/transcripts/.

Deux traitements au passage :

- **Scrubbing** des clés API (`sk-or-…`, `sk-ant-…`). Une clé OpenRouter est
  apparue dans un affichage de ~/.bash_history pendant le stage ; elle est donc
  présente dans au moins un transcript. Le script refuse d'écrire un fichier
  où il resterait une clé après nettoyage.
- **Troncature des sorties d'outils** à --max-tool-chars (défaut 2000). Les
  transcripts bruts font 55 Mo, dont l'essentiel est du log de commande sans
  valeur pour reprendre le travail. Le texte des échanges, lui, est conservé
  intégralement.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SRC = Path.home() / ".claude/projects/-home-anass-GenerativePDE"
DST = Path(__file__).resolve().parent / "transcripts"

SECRET = re.compile(r"sk-(?:or|ant)-[A-Za-z0-9_-]{16,}")


def scrub(text: str) -> str:
    return SECRET.sub("[CLE-API-SUPPRIMEE]", text)


def blocks(content) -> list[str]:
    """Aplatit le champ `content` d'un message en une liste de morceaux texte."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", ""))
        elif t == "thinking":
            out.append("*(réflexion)*\n\n" + b.get("thinking", ""))
        elif t == "tool_use":
            inp = json.dumps(b.get("input", {}), ensure_ascii=False, indent=2)
            out.append(f"**Outil : {b.get('name','?')}**\n```json\n{inp}\n```")
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c
                              if isinstance(x, dict) and x.get("type") == "text")
            out.append(f"**Résultat**\n```\n{c}\n```")
    return out


def convert(path: Path, max_tool: int) -> tuple[Path, int, int]:
    lines_out, n_msg = [f"# Transcript `{path.stem}`\n"], 0
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            parts = [p for p in blocks(msg.get("content")) if p and p.strip()]
            if not parts:
                continue
            n_msg += 1
            who = "Utilisateur" if role == "user" else "Claude"
            ts = rec.get("timestamp", "")
            lines_out.append(f"\n## {who}{f'  ·  {ts}' if ts else ''}\n")
            for p in parts:
                # Les sorties d'outils sont tronquées, pas le texte des échanges.
                if p.startswith(("**Résultat**", "**Outil :")) and len(p) > max_tool:
                    p = p[:max_tool] + f"\n… [tronqué, {len(p) - max_tool} caractères]"
                lines_out.append(p + "\n")

    text = scrub("\n".join(lines_out))
    leftover = len(SECRET.findall(text))
    if leftover:
        raise SystemExit(f"ABANDON : {leftover} clé(s) subsistent dans {path.name}")
    out = DST / f"{path.stem}.md"
    out.write_text(text, encoding="utf-8")
    return out, n_msg, len(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tool-chars", type=int, default=2000)
    args = ap.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"introuvable : {SRC}")
    DST.mkdir(exist_ok=True)

    total = 0
    for f in sorted(SRC.glob("*.jsonl")):
        out, n, size = convert(f, args.max_tool_chars)
        total += size
        print(f"  {out.name:42}  {n:5} messages  {size/1024:8.0f} Ko")
    print(f"\n{total/1024/1024:.1f} Mo écrits dans {DST}")


if __name__ == "__main__":
    main()
