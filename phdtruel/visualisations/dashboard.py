"""Streamlit dashboard: browse generated figures and copy LaTeX (including subfigures)."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import phdtruel

import streamlit as st

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".svg",
        ".eps",
        ".tiff",
        ".tif",
        ".bmp",
    }
)
_config_figures_dir = phdtruel.config.get("figures_dir")
DEFAULT_FIGURES_DIR = (
    Path(_config_figures_dir)
    if _config_figures_dir is not None
    else Path.home() / "data" / "generated_figures"
)


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


@st.cache_data(show_spinner=False)
def list_images_cached(root_s: str) -> tuple[str, ...]:
    root = Path(root_s)
    return tuple(str(p) for p in _list_images(root))


def _escape_latex(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("#", r"\#")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("_", r"\_")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def _latex_path_for_graphic(full: Path, base: Path, prefix: str) -> str:
    try:
        rel = full.resolve().relative_to(base.resolve())
    except ValueError:
        rel = Path(full.name)
    parts: list[str] = []
    if prefix.strip():
        for part in Path(prefix).parts:
            parts.append(part)
    for part in rel.parts:
        parts.append(part)
    out_parts: list[str] = []
    for part in parts:
        part_esc = part.replace("#", r"\#").replace("%", r"\%")
        part_esc = part_esc.replace(" ", r"\ ")
        part_esc = part_esc.replace("_", r"\_")
        out_parts.append(part_esc)
    return "/".join(out_parts)


def build_latex_figure(
    rel_paths: list[str],
    per_captions: list[str] | None,
    main_caption: str,
    main_label: str,
    cols: int,
    width_mode: str,
) -> str:
    n = len(rel_paths)
    if n == 0:
        return r"% (no images selected — pick at least one file above.)"

    per_captions = per_captions or [""] * n
    while len(per_captions) < n:
        per_captions.append("")

    key = re.sub(r"[^a-zA-Z0-9:-]", "-", main_label) if main_label else "fig"
    main_cap = _escape_latex(main_caption.strip()) or "Main caption"
    c = max(1, min(cols, 6))

    if width_mode == "equal row":
        w_sub = f"{(0.98 / c):.3f}\\textwidth"
        w_inc = w_sub
    else:
        w_sub = r"0.98\linewidth"
        w_inc = w_sub

    if n == 1:
        w1 = r"0.9\textwidth" if width_mode == "equal row" else r"0.98\linewidth"
        rp = rel_paths[0]
        return "\n".join(
            [
                r"\begin{figure}[ht]",
                r"  \centering",
                f"  \\includegraphics[width={w1}]{{{rp}}}",
                f"  \\caption{{{main_cap}}}",
                f"  \\label{{fig:{key}}}",
                r"\end{figure}",
            ]
        )

    lines: list[str] = [r"\begin{figure}[ht]", r"  \centering"]
    for i, (rp, subcap) in enumerate(zip(rel_paths, per_captions, strict=True)):
        sub_label = f"{key}_sub{chr(ord('a') + i)}"
        if subcap.strip():
            cap = _escape_latex(subcap.strip())
        else:
            fname = Path(rp).name
            cap = f"\\texttt{{{_escape_latex(fname)}}}"
        lines.append(f"  \\begin{{subfigure}}[b]{{{w_sub}}}")
        lines.append(f"    \\includegraphics[width={w_inc}]{{{rp}}}")
        lines.append(f"    \\caption{{{cap}}}")
        lines.append(f"    \\label{{fig:{sub_label}}}")
        lines.append("  \\end{subfigure}")
        if (i + 1) % c == 0 and (i + 1) < n:
            lines.append(r"  \\[0.5em]")
        elif (i + 1) < n:
            lines.append(r"  \hfill")

    lines.append(f"  \\caption{{{main_cap}}}")
    lines.append(f"  \\label{{fig:{key}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def _preview_svg(svg_path: Path) -> None:
    b64 = base64.b64encode(svg_path.read_bytes()).decode("ascii")
    st.markdown(
        f'<div style="max-width:100%;overflow:auto"><img src="data:image/svg+xml;base64,{b64}" /></div>',
        unsafe_allow_html=True,
    )
    st.caption(svg_path.name)


def preview_image(p: Path) -> None:
    ext = p.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp"}:
        st.image(str(p), use_container_width=True)
    elif ext == ".svg":
        try:
            _preview_svg(p)
        except OSError:
            st.caption("Could not read SVG file.")
    elif ext == ".pdf":
        st.info(
            f"PDF preview is not inline. File: `{p}` — "
            "LaTeX `\\includegraphics` works with `pdflatex`."
        )
    elif ext == ".eps":
        st.info(
            "EPS: preview skipped; `\\includegraphics` with `pdflatex` or `eps` drivers."
        )
    else:
        st.caption(f"Unusual format {ext!r} for preview.")


def _toggle_selected(path_s: str) -> None:
    selected = st.session_state.get("selected_images", [])
    if path_s in selected:
        selected = [s for s in selected if s != path_s]
    else:
        selected = [*selected, path_s]
    st.session_state["selected_images"] = selected


def main() -> None:
    st.set_page_config(
        page_title="Figure dashboard",
        page_icon="📊",
        layout="wide",
    )
    st.title("Generated figures & LaTeX")
    st.caption(
        "Scan a folder, filter files, then select one or more images in order for subfigures (a)–(b)–(c)…"
    )

    with st.sidebar:
        base_str = st.text_input(
            "Figures root directory",
            value=str(DEFAULT_FIGURES_DIR),
            help="Recursively listed.",
        )
        base = Path(base_str).expanduser()
        search = st.text_input("Filter (substring in relative path)", value="")
        top_options: list[str | None] = [None]
        if base.is_dir():
            br = base.resolve()
            tops: set[str] = set()
            for s in list_images_cached(str(br)):
                p = Path(s)
                try:
                    rel = p.relative_to(br)
                except ValueError:
                    continue
                if rel.parts:
                    tops.add(rel.parts[0])
            top_options.extend(sorted(tops))
        top_label = "All subfolders"
        top_idx = st.selectbox(
            "Limit to top-level subfolder",
            options=range(len(top_options)),
            format_func=lambda i: top_label
            if top_options[i] is None
            else str(top_options[i]),
        )
        top_only = top_options[top_idx]
        exts = st.multiselect(
            "Extensions",
            options=sorted(IMAGE_EXTENSIONS),
            default=sorted(
                e
                for e in (".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg")
                if e in IMAGE_EXTENSIONS
            ),
        )
        tex_prefix = st.text_input(
            "Optional path prefix in LaTeX (POSIX, no leading slash needed)",
            value="",
            help="E.g. `figs/genetated` if your .tex has `figs/genetated/...` under the project root.",
        )
        st.divider()
        st.subheader("Layout")
        cols = st.number_input(
            "Subfigures per row", min_value=1, max_value=6, value=2, step=1
        )
        width_mode = st.radio(
            "Width", options=["equal row", "full line"], index=0, horizontal=True
        )
        st.divider()
        main_caption = st.text_input("Main figure caption (optional)", value="")
        main_label = st.text_input("Label key (no `fig:` prefix)", value="res")

    if not base.is_dir():
        st.error(f"Not a directory: {base}")
        st.stop()

    base_r = base.resolve()
    all_paths = [Path(s) for s in list_images_cached(str(base_r))]
    exts_l = {e.lower() for e in exts} if exts else IMAGE_EXTENSIONS
    filtered: list[Path] = []
    for p in all_paths:
        if p.suffix.lower() not in exts_l:
            continue
        try:
            rel = p.relative_to(base_r)
        except ValueError:
            rel = p
        srel = str(rel)
        if top_only is not None and (not rel.parts or rel.parts[0] != top_only):
            continue
        if search and search.lower() not in srel.lower():
            continue
        filtered.append(p)

    st.caption(
        f"Showing {len(filtered)} file(s) — {len(all_paths)} total in tree under `{base_r}`"
    )

    options = [str(p) for p in filtered]
    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = []
    st.session_state["selected_images"] = [
        s for s in st.session_state["selected_images"] if s in options
    ]

    if top_only is not None and options:
        st.subheader(f"Preview in `{top_only}`")
        preview_limit = st.number_input(
            "Preview items shown",
            min_value=6,
            max_value=120,
            value=24,
            step=6,
            help="Click Select/Unselect under each preview.",
        )
        preview_count = min(int(preview_limit), len(filtered))
        gallery = st.columns(4)
        for i, p in enumerate(filtered[:preview_count]):
            path_s = str(p)
            picked = path_s in st.session_state["selected_images"]
            with gallery[i % 4]:
                st.caption(p.name)
                preview_image(p)
                st.button(
                    "Unselect" if picked else "Select",
                    key=f"pick_{path_s}",
                    on_click=_toggle_selected,
                    args=(path_s,),
                    use_container_width=True,
                )
        if len(filtered) > preview_count:
            st.caption(
                f'Showing {preview_count}/{len(filtered)} previews. Increase "Preview items shown" to see more.'
            )

    selected = st.multiselect(
        "Selected images (order = subfigure a, b, c …; pick in the order you want.)",
        options=options,
        default=st.session_state["selected_images"],
        key="selected_images",
    )

    per_caps: list[str] = []
    if selected:
        st.subheader("Sub-captions (optional, defaults to filename stem)")
        grid = st.columns(min(3, max(1, len(selected))))
        for i, sp in enumerate(selected):
            with grid[i % len(grid)]:
                name = Path(sp).name
                per_caps.append(
                    st.text_input(
                        f"({chr(ord('a') + i)}) {name}",
                        value=Path(name).stem.replace("_", " "),
                        key=f"subcap_{i}",
                    )
                )

    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("LaTeX to paste in your .tex file")
    with c2:
        include_packages = st.checkbox(
            "Prepend `graphicx` + `subcaption` comments", value=True
        )

    rel_paths = (
        [_latex_path_for_graphic(Path(s), base_r, tex_prefix) for s in selected]
        if selected
        else []
    )

    body = build_latex_figure(
        rel_paths,
        per_caps if selected else None,
        main_caption,
        main_label,
        int(cols),
        str(width_mode),
    )
    if include_packages and selected:
        body = (
            r"% \usepackage{graphicx}"
            + "\n"
            + r"% \usepackage{subcaption}"
            + "\n\n"
            + body
            if len(selected) > 1
            else r"% \usepackage{graphicx}" + "\n\n" + body
        )
    elif include_packages and not selected:
        body = r"% \usepackage{graphicx}" + "\n\n" + body

    st.code(body, language="latex", line_numbers=True)
    st.text_area(
        "Copy from here (Ctrl+A, Ctrl+C)", value=body, height=min(28 * 15, 400)
    )

    st.divider()
    st.subheader("Preview of selection (max 12)")
    for sp in selected[:12]:
        p = Path(sp)
        st.markdown(f"**`{p.name}`** — `{p.parent}`")
        preview_image(p)
        st.divider()
    if len(selected) > 12:
        st.caption("Only the first 12 are previewed.")


if __name__ == "__main__":
    main()
