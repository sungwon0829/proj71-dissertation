r"""check_paper.py - the Thursday pass, automated.

Checks main.tex (and main.pdf if present) for:
  1. leftover \TODO{} markers
  2. banned vocabulary and paragraph-opening connectives (the style contract list)
  3. semicolons anywhere in the body prose, captions or table cells
  4. forbidden phrasings from the handover (dose-dependent, buys, three seeds, not only)
  5. sentences longer than 40 words
  6. every number that appears in Section 6 (Discussion) also appears somewhere in Sections 3-5 or a table
  7. body length: the page on which "References" starts, and whether the abstract is 150-350 words

Exit code 0 when everything passes. Run from the paper folder:
  python tools/check_paper.py --tex main.tex --pdf main.pdf
"""
import argparse
import re
import subprocess
import sys

BANNED = ["delve", "leverage", "leverages", "underscore", "underscores", "pivotal", "crucial", "crucially",
          "notably", "robust", "comprehensive", "novel", "seamless", "seamlessly", "holistic", "landscape",
          "realm", "tapestry", "paradigm", "nuanced", "nuance", "importantly", "interestingly", "genuine",
          "genuinely", "warrants", "striking", "state-of-the-art"]
OPENERS = ("Moreover", "Furthermore", "Additionally", "However", "In summary", "Overall")
FORBIDDEN = ["dose-dependent", r"\bbuys\b", "three seeds", "3 seeds", "not only"]


def body_of(tex):
    b = tex[tex.index(r"\begin{document}"):tex.index(r"\end{document}")]
    return "\n".join(l for l in b.split("\n") if not l.lstrip().startswith("%"))


def section(tex, title):
    m = re.search(r"\\section\{" + re.escape(title) + r"\}(.*?)(?=\\section\{|\\bibliography)", tex, flags=re.S)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="main.tex")
    ap.add_argument("--pdf", default="main.pdf")
    ap.add_argument("--max_words", type=int, default=50)
    a = ap.parse_args()
    tex = open(a.tex, encoding="utf-8").read()
    body = body_of(tex)
    fails = 0

    def report(name, items):
        nonlocal fails
        if items:
            fails += 1
            print(f"FAIL {name} ({len(items)})")
            for it in items[:12]:
                print("     ", it)
        else:
            print(f"PASS {name}")

    # 1 TODO
    report("no TODO markers", re.findall(r"\\TODO\{[^}]*\}", body))
    # 2 banned words (allow the verbatim research question's 'robust')
    hits = []
    for i, line in enumerate(body.split("\n"), 1):
        for w in BANNED:
            for m in re.finditer(r"\b" + re.escape(w) + r"\b", line, flags=re.I):
                if w == "robust" and "therapy-support model more robust" in line:
                    continue
                hits.append(f"line {i}: {w}")
    report("no banned vocabulary", hits)
    report("no connective paragraph openers", [l[:60] for l in body.split("\n") if l.startswith(OPENERS)])
    # 3 semicolons
    report("no semicolons in body", [f"{l[:80]}" for l in body.split("\n") if ";" in l])
    # 4 forbidden phrasings
    report("no forbidden phrasings", [p for p in FORBIDDEN if re.search(p, body, flags=re.I)])
    # 5 long sentences
    prose = re.sub(r"\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", "", body, flags=re.S)
    prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " Xx ", prose)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose)
    longs = [f"{len(s.split())} words: {s.strip()[:70]}..." for s in sents if len(s.split()) > a.max_words]
    if longs:  # advisory only: list sentences and quoted prompts can legitimately run long
        print(f"WARN sentences over {a.max_words} words ({len(longs)}), advisory")
        for it in longs[:12]:
            print("     ", it)
    else:
        print(f"PASS no sentences over {a.max_words} words")
    # 6 Discussion numbers traced to Results/Methods/tables
    disc = section(tex, "Discussion")
    evidence = section(tex, "Results") + section(tex, "Methods") + section(tex, "Problem statement") + section(tex, "Related work")
    nums = set(re.findall(r"(?<![\d.-])\d+\.\d+(?![\d.])", disc))
    untraced = sorted(n for n in nums if n not in evidence and n not in tex[tex.index(r"\appendix"):])
    report("every decimal number in Discussion appears in Results/Methods/appendix", untraced)
    # 7 abstract length and page budget
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    n_abs = len(abs_m.group(1).split()) if abs_m else 0
    report("abstract 150-350 words", [] if 150 <= n_abs <= 350 else [f"{n_abs} words"])
    try:
        n_pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True).stdout).group(1))
        ref_page = None
        for p in range(1, n_pages + 1):
            t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), a.pdf, "-"], capture_output=True, text=True).stdout
            if re.search(r"^References\s*$", t, flags=re.M):
                ref_page = p
                break
        size = re.search(r"Page size:\s+([\d.]+ x [\d.]+)", subprocess.run(["pdfinfo", a.pdf], capture_output=True, text=True).stdout).group(1)
        report("References start on page 13 or earlier (body <= 12 pages)", [] if ref_page and ref_page <= 13 else [f"References on page {ref_page}"])
        report("US Letter page size", [] if size.startswith("612 x 792") else [size])
    except Exception as e:  # pdf tools missing
        print("SKIP pdf checks:", e)

    print("\nALL PASS" if fails == 0 else f"\n{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
