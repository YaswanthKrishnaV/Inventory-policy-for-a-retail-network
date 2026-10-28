"""Build memo_A.pdf / memo_B.pdf from memo/memo_<X>.md (pandoc -> HTML -> wkhtmltopdf).

    python memo/build_memo.py          # both memos
    python memo/build_memo.py B        # one memo
The memo text is written from the generated results (reports/results_<cutoff>.md for A,
reports/project_b/results.md for B); any tool chain that renders the markdown works.
"""
import pathlib, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
root = here.parent
css = here / "memo.css"
for x in (sys.argv[1:] or ["A", "B"]):
    html = here / f"memo_{x}.html"
    subprocess.run(["pandoc", str(here / f"memo_{x}.md"), "-s", "--metadata", "title=Memo", "-c", str(css), "-o", str(html)], check=True)
    subprocess.run(["wkhtmltopdf", "--enable-local-file-access", "--disable-smart-shrinking", "-q", "-s", "A4",
                    "-T", "12mm", "-B", "12mm", "-L", "14mm", "-R", "14mm", str(html), str(root / f"memo_{x}.pdf")], check=True)
    print("wrote", root / f"memo_{x}.pdf")
