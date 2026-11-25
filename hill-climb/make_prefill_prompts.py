"""Build natural-text prompts of exact token lengths for prefill benching."""
import re, sys, pathlib

sys.path.insert(0, r"C:\LLM models\project-maya\tools")
from strata_tokenizer import Tokenizer

GGUF = r"C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf"
OUT = pathlib.Path(r"C:\LLM models\project-maya\hill-climb")
SRC = [r"C:\LLM models\project-maya\docs\INKLING2.md",
       r"C:\LLM models\project-maya\docs\SETUP.md",
       r"C:\LLM models\project-maya\docs\MODELS.md"]

def clean(t: str) -> str:
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # drop code blocks
    t = re.sub(r"`([^`]+)`", r"\1", t)                    # inline code -> text
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)           # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)        # links -> label
    t = re.sub(r"[#*_>|]+", " ", t)                       # md decoration
    t = re.sub(r"\s+", " ", t)
    return t.strip()

tk = Tokenizer.from_gguf(GGUF)
text = " ".join(clean(pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")) for p in SRC)
ids = tk.encode(text)
print(f"source tokens: {len(ids)}")

for n in (512, 2048, 8192):
    assert len(ids) >= n, f"not enough tokens for {n}"
    (OUT / f"prefill-tokens-{n}.txt").write_text(",".join(map(str, ids[:n])) + "\n", encoding="ascii")
    back = tk.decode(ids[:n])
    print(f"wrote {n} ids; roundtrip head: {back[:80]!r}")
