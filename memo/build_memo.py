"""Build memo_A.pdf from memo/memo_A.md (pandoc -> HTML -> wkhtmltopdf).

    python memo/build_memo.py
Any tool chain that renders the markdown works; the numbers in the memo come from
reports/results_<cutoff>.md.
"""
import pathlib, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
root = here.parent
css = here / "memo.css"
html = here / "memo_A.html"
subprocess.run(["pandoc", str(here / "memo_A.md"), "-s", "--metadata", "title=Memo", "-c", str(css), "-o", str(html)], check=True)
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--disable-smart-shrinking", "-q", "-s", "A4", "-T", "12mm", "-B", "12mm", "-L", "14mm", "-R", "14mm",
                str(html), str(root / "memo_A.pdf")], check=True)
print("wrote", root / "memo_A.pdf")
