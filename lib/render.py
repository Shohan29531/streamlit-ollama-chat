import re
import streamlit as st

RE_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
RE_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)

def normalize_math(text: str) -> str:
    # \[...\] -> $$...$$
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # \(...\) -> $...$
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # (\displaystyle ... ) -> $...$
    text = re.sub(r"\(\s*\\displaystyle\s*(.+?)\s*\)", r"$\\displaystyle \1$", text, flags=re.DOTALL)

    # [ ... ] that looks like math -> $$...$$
    def bracket_repl(m):
        inner = m.group(1).strip()
        looks_math = (
            "\\" in inner or "_{" in inner or "^{" in inner or
            r"\frac" in inner or r"\binom" in inner or
            r"\sum" in inner or r"\int" in inner or
            r"\ge" in inner or r"\le" in inner or
            r"\text{" in inner
        )
        return f"$$\n{inner}\n$$" if looks_math else m.group(0)

    text = re.sub(r"(?<!\w)\[(.+?)\](?!\w)", bracket_repl, text, flags=re.DOTALL)
    return text

def render_chat_text(text: str) -> None:
    text = normalize_math(text)

    # Render $$...$$ blocks via st.latex
    parts = RE_BLOCK.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            st.latex(part.strip())
            continue

        # Render inline $...$ via st.latex too (reliable)
        subparts = RE_INLINE.split(part)
        for j, sp in enumerate(subparts):
            if j % 2 == 1:
                st.latex(sp.strip())
            else:
                if sp.strip():
                    st.markdown(sp)
