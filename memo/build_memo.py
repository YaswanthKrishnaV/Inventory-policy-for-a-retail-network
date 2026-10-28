"""Build memo_B.pdf from memo/memo_B.md (pandoc -> HTML -> wkhtmltopdf).

    python memo/build_memo.py
Any tool chain that renders the markdown works; the numbers in the memo come from
reports/results_<cutoff>.md.
"""
import pathlib, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
root = here.parent
css = here / "memo.css"
html = here / "memo_B.html"
subprocess.run(["pandoc", str(here / "memo_B.md"), "-s", "--metadata", "title=Memo", "-c", str(css), "-o", str(html)], check=True)
subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--disable-smart-shrinking", "-q", "-s", "A4", "-T", "12mm", "-B", "12mm", "-L", "14mm", "-R", "14mm",
                str(html), str(root / "memo_B.pdf")], check=True)
print("wrote", root / "memo_B.pdf")
