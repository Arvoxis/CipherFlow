"""Build the CipherFlow Invention Disclosure Form (IDF-B) as .docx.

Follows the VIT IPR&TTCELL IDF-B template: the same ten numbered sections, the same running
header, the same closing marker. Everything is generated, so the document can be rebuilt after
any number changes rather than hand-edited into drift.

Formatting rules enforced here, because the hand-built version broke all of them:
  * every table column has an explicit width and the widths sum to the text width, so no cell
    can push its text past the border
  * table body font is sized to the column it sits in, so words wrap instead of breaking
    mid-character
  * section headings carry keep_with_next, so a heading cannot be orphaned at a page foot
  * table rows carry cantSplit and header rows repeat, so a row cannot be torn across pages
  * every image is scaled to the text width and never exceeds it

Run:
    python scripts/make_idf.py
    python scripts/make_idf.py --pdf      # also export PDF via Word COM (Windows only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
ART = REPO_ROOT / "artifacts"
FIGS = ART / "figures"
SHOTS = ART / "screenshots"
OUT_DOCX = REPO_ROOT / "CipherFlow_IDF.docx"

# Typography lifted from the VIT IPR&TTCELL reference form, measured span by span:
#   body            Calibri 12 pt, black
#   section heading Calibri Bold 12 pt, black (not coloured)
#   table body      Calibri 10 pt, black
#   table header    Calibri Bold 10 pt, black, with no cell shading of any kind
#   form title      Times New Roman Bold 12 pt
#   running header  Times New Roman 10 pt
#   bullets         Times New Roman filled circle at 1.0 in, text at 1.1 in
#   TRL tick        Wingdings check
# Letter page with 0.75 in side margins, giving 7.0 in of usable width.
TEXT_W = 7.0
BODY_FONT = "Calibri"
SERIF_FONT = "Times New Roman"
BODY_PT = 12.0
HEAD_PT = 12.0
TABLE_PT = 10.0
BLACK = RGBColor(0x00, 0x00, 0x00)
HEAD_RGB = BLACK

# Technology readiness level claimed for this invention: validated in a laboratory environment.
TRL_LEVEL = 4


# --------------------------------------------------------------------------- low-level helpers
def _set_cell_width(cell, inches: float) -> None:
    """python-docx needs the width on every cell, not just the column, or Word re-flows it."""
    cell.width = Inches(inches)
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:tcW")):
        tcPr.remove(old)
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(int(inches * 1440)))
    tcW.set(qn("w:type"), "dxa")
    tcPr.append(tcW)


def _force_font(run, name: str) -> None:
    """Pin a run to one font family on every script slot.

    Setting run.font.name alone only writes w:ascii. Word then resolves a symbol font such as
    Wingdings through its own fallback and the glyph comes out as a plain letter, which is why
    the tick has to be nailed down explicitly.
    """
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), name)


def _shade(cell, hex_fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(shd)


def _no_split(row) -> None:
    """Stop Word tearing a row in half across a page break."""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    trPr.append(el)


def _repeat_header(row) -> None:
    """Repeat this row at the top of every page the table spans."""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _cell_text(cell, text: str, size: float, bold=False, align=None, color=None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    run.font.name = BODY_FONT
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color


def make_table(doc, rows: int, cols: int, widths: list[float]):
    t = doc.add_table(rows=rows, cols=cols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")  # honour our widths instead of auto-fitting
    tblPr.append(layout)
    for r in t.rows:
        _no_split(r)
        for c, w in zip(r.cells, widths, strict=True):
            _set_cell_width(c, w)
    return t


def heading(doc, text: str, size=HEAD_PT, space_before=12):
    """A numbered section heading that will not be left stranded at the foot of a page."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(5)
    pf.keep_with_next = True  # the fix for "headings not placed correctly"
    run = p.add_run(text)
    run.font.name = BODY_FONT
    run.font.size = Pt(size)
    run.bold = True
    run.font.color.rgb = HEAD_RGB
    return p


def sub(doc, text: str, size=BODY_PT):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = BODY_FONT
    r.font.size = Pt(size)
    r.bold = True
    return p


def body(doc, text: str, size=BODY_PT, space_after=6, italic=False, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.name = BODY_FONT
    r.font.size = Pt(size)
    r.italic = italic
    return p


def bullets(doc, items: list[str], size=BODY_PT):
    """Filled-circle bullets drawn exactly as the reference form does them: a Times New Roman
    glyph at 1.0 in with the text at 1.1 in, rather than Word's own List Bullet style."""
    for it in items:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Inches(1.1 - 0.75)  # measured from the reference: glyph 1.0in, text 1.1in
        pf.first_line_indent = Inches(-0.1)
        pf.space_after = Pt(3)
        g = p.add_run("●  ")
        g.font.name = SERIF_FONT
        g.font.size = Pt(size)
        r = p.add_run(it)
        r.font.name = BODY_FONT
        r.font.size = Pt(size)


def numbered(doc, items: list[str], size=BODY_PT):
    """Manual numbering. Word's List Number style continues across every list in a document,
    which is what produced the 16, 17, 18 numbering in the previous build."""
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"{i}. {it}")
        r.font.name = BODY_FONT
        r.font.size = Pt(size)


def figure(doc, path: Path, caption: str, width: float | None = None, max_h: float = 7.2):
    """Place an image scaled to the text width, never taller than the usable page height."""
    if not path.exists():
        print(f"  !! missing image: {path}")
        return
    from PIL import Image

    w_px, h_px = Image.open(path).size
    w = width or TEXT_W
    if w * h_px / w_px > max_h:  # too tall once scaled to width, so bound by height instead
        w = max_h * w_px / h_px
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(w))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    r = c.add_run(caption)
    r.font.name = BODY_FONT
    r.font.size = Pt(10)
    r.italic = True
    r.font.color.rgb = BLACK


def data_table(
    doc,
    header: list[str],
    rows: list[list[str]],
    widths: list[float],
    font=TABLE_PT,
    head_font=TABLE_PT,
    bold_rows: list[int] | None = None,
):
    """The reference form shades nothing: plain black borders, black text, white cells. A row
    worth drawing the eye to is set bold rather than filled."""
    t = make_table(doc, len(rows) + 1, len(header), widths)
    for j, h in enumerate(header):
        _cell_text(t.rows[0].cells[j], h, head_font, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLACK)
    _repeat_header(t.rows[0])
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            al = WD_ALIGN_PARAGRAPH.CENTER if j and _numeric(v) else None
            _cell_text(t.rows[i + 1].cells[j], v, font, bold=bool(bold_rows and i in bold_rows), align=al)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def _numeric(v: str) -> bool:
    return (
        str(v).replace(".", "").replace("-", "").replace("+", "").replace("%", "").replace(",", "").isdigit()
    )


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# --------------------------------------------------------------------------- the document
def build() -> Document:
    doc = Document()

    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(BODY_PT)

    # Margins measured off the reference: body text starts at 0.75 in on both sides.
    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = Inches(0.75)
        s.top_margin = Inches(0.5)
        s.bottom_margin = Inches(0.75)
        hp = s.header.paragraphs[0]
        hp.text = ""
        hr = hp.add_run("©VIT IPR&TTCELL")
        hr.font.name = SERIF_FONT
        hr.font.size = Pt(10)

    # ------------------------------------------------------------------ title block
    # The reference sets the form title in Times New Roman Bold 12, not a large sans face.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("Invention Disclosure Format (IDF)-B")
    r.font.name = SERIF_FONT
    r.font.size = Pt(12)
    r.bold = True

    # "Document No" and friends are Arial 12 in the reference.
    meta = make_table(doc, 3, 2, [3.5, 3.5])
    for i, (k, v) in enumerate([("Document No", ""), ("Issue No/Date", ""), ("Amd. No/Date", "")]):
        for col, val in ((0, k), (1, v)):
            _cell_text(meta.rows[i].cells[col], val, 12)
            for run in meta.rows[i].cells[col].paragraphs[0].runs:
                run.font.name = "Arial"
    doc.add_paragraph()

    # ------------------------------------------------------------------ 1. Title
    heading(doc, "1.  Title of the invention:", space_before=6)
    body(
        doc,
        "Method and System for Payload-Free Classification of Encrypted Network Traffic "
        "Using Self-Supervised Flow-Shape Tokenisation (CipherFlow).",
    )

    inv = make_table(doc, 2, 2, [1.7, 5.3])
    _cell_text(inv.rows[0].cells[0], "Inventors", TABLE_PT, bold=True)
    _cell_text(inv.rows[0].cells[1], "Rakshit Sinha; Suyash Dabholkar", TABLE_PT)
    _cell_text(inv.rows[1].cells[0], "Institution", TABLE_PT, bold=True)
    _cell_text(inv.rows[1].cells[1], "Vellore Institute of Technology", TABLE_PT)
    doc.add_paragraph()

    # ------------------------------------------------------------------ 2. Field
    heading(doc, "2.  Field /Area of invention:")
    body(doc, "This invention belongs to the fields of:")
    bullets(
        doc,
        [
            "Network Traffic Analysis and Encrypted Traffic Classification",
            "Machine Learning, Deep Learning and Self-Supervised Representation Learning",
            "Cyber Security, Intrusion Detection and Command-and-Control Traffic Detection",
            "Network Monitoring Apparatus and Middlebox Systems",
        ],
    )
    body(
        doc,
        "The proposed system allows a network monitoring apparatus to identify the application "
        "behind an encrypted flow, and to flag malicious command-and-control activity, without "
        "decrypting the flow and without reading or storing any packet payload. It observes only "
        "the shape of a flow: the size of each packet, the time between packets, and the "
        "direction each packet travelled. Because none of those three quantities is hidden by "
        "encryption, the apparatus keeps working on TLS 1.3 and QUIC traffic, where conventional "
        "deep packet inspection returns nothing at all.",
    )

    # ------------------------------------------------------------------ 3. Prior art
    # Start the table on a fresh page. Left where it was, only the header row fitted at the foot
    # of page 1 and every data row landed overleaf.
    page_break(doc)
    heading(
        doc,
        "3.  Prior Patents and Publications from literature (provide a table summarizing the prior art)",
        space_before=0,
    )
    prior_head = [
        "No.",
        "Title",
        "Type",
        "Authors / Inventors",
        "Venue / No.",
        "Year",
        "Key Features",
        "Limitations / Gaps",
        "Differentiating Methodology",
    ]
    prior_w = [0.33, 1.06, 0.50, 0.76, 0.70, 0.44, 1.02, 1.08, 1.11]
    prior_rows = [
        [
            "1",
            "ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers "
            "for Encrypted Traffic Classification",
            "Paper",
            "Lin, Xu, Liu, Wang et al.",
            "WWW",
            "2022",
            "Applies BERT-style masked pre-training to encrypted traffic. Learns contextual "
            "representations of datagrams and fine-tunes for traffic classification.",
            "Tokenises payload / datagram byte n-grams, so it depends on observable payload bytes. "
            "Fails once traffic is fully encrypted and is defeated cheaply by byte-level "
            "obfuscation or re-encoding.",
            "Ours tokenises flow SHAPE only, that is quantised packet size, inter-arrival time and "
            "direction, and reads zero payload bytes. The masking objective is defined over that "
            "shape alphabet, so pre-training remains possible when no bytes are readable.",
        ],
        [
            "2",
            "Deep Packet: A Novel Approach for Encrypted Traffic Classification Using Deep Learning",
            "Paper",
            "Lotfollahi, Jafari Siavoshani, et al.",
            "Soft Comput.",
            "2020",
            "Feeds raw packet bytes to a CNN and a stacked autoencoder to classify applications "
            "and traffic characterisation categories.",
            "Payload dependent and fully supervised. Requires large labelled corpora and degrades "
            "as encryption coverage increases.",
            "Ours needs no payload and no large labelled corpus. Pre-training on unlabelled flows "
            "supplies the representation; only a small labelled set fits the head.",
        ],
        [
            "3",
            "FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition",
            "Paper",
            "Shapira & Shavitt",
            "IEEE INFOCOM WKSHPS",
            "2019",
            "Renders a flow as a two-dimensional size/time histogram image and classifies it with "
            "an image CNN. Uses no payload.",
            "Aggregating into a histogram destroys packet ORDER. Fully supervised, with no "
            "self-supervised pre-training and no transferable representation.",
            "Ours keeps the flow as an ordered token sequence, so request-response structure is "
            "preserved, and it learns the representation without labels.",
        ],
        [
            "4",
            "nPrint / nPrintML: Automating Traffic Analysis with a Standard Packet Representation",
            "Paper",
            "Holland, Schmitt, Feamster, Mittal",
            "ACM CCS",
            "2021",
            "Standard per-packet header bitmap representation fed to automated machine learning "
            "pipelines for many traffic analysis tasks.",
            "Driven by header fields and supervised end to end. No masked self-supervised "
            "objective over a quantised shape alphabet, and no evasion-robust training.",
            "Ours defines a compact quantised alphabet over three observable quantities, pre-trains "
            "on it by masked reconstruction, and trains explicitly against padding and jitter.",
        ],
        [
            "5",
            "Flow-statistic classifiers (CICFlowMeter / nfstream features with RandomForest or XGBoost)",
            "Class",
            "Various",
            "Deployed",
            "2016+",
            "Compute aggregate per-flow statistics such as mean and variance of packet size, "
            "duration and byte counts, then train a supervised tree ensemble.",
            "Aggregation discards temporal order, so flows differing only in structure are "
            "indistinguishable. Labels are required throughout and nothing transfers between tasks. "
            "Collapses under packet padding and timing jitter.",
            "Measured here: these baselines sit near chance on order-defined classes where ours "
            "reaches 0.9944, and they lose 17.7% of their accuracy to evasion where our augmented "
            "model loses none.",
        ],
        [
            "6",
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "Paper",
            "Devlin, Chang, Lee, Toutanova",
            "NAACL-HLT",
            "2019",
            "Establishes masked-token self-supervised pre-training over a discrete vocabulary, "
            "later fine-tuned for downstream tasks.",
            "Defined over natural-language tokens. It does not address network traffic, payload-free "
            "observation, or adversarial reshaping of the input.",
            "Ours supplies the missing ingredient for traffic: a discrete, ordered, payload-free "
            "alphabet derived from flow shape, over which a masked objective can be defined at all.",
        ],
    ]
    data_table(doc, prior_head, prior_rows, prior_w, font=9.0, head_font=9.0)
    body(
        doc,
        "Note: the references above are from the inventors' own review of the research "
        "literature. A professional patent-literature search has not yet been commissioned and "
        "is listed as an outstanding item before filing.",
        size=10,
        italic=True,
    )

    # ------------------------------------------------------------------ 4. Summary / gap
    page_break(doc)
    heading(doc, "4.  Summary and background of the invention (Address the gap / Novelty)", space_before=0)

    sub(doc, "Background of the Invention -")
    body(
        doc,
        "Modern transport security has removed the very information that traffic-inspection "
        "equipment was built to read. TLS 1.3 encrypts the certificate exchange and collapses "
        "the handshake, and QUIC carries almost everything inside an encrypted packet. Deep "
        "packet inspection matches patterns against payload bytes, so against either protocol "
        "it returns nothing useful.",
    )
    body(
        doc,
        "An operator who still needs to know what is crossing the network is left with three "
        "poor options. The first is to terminate TLS at a middlebox, which means installing a "
        "certificate authority on every endpoint and creating a decryption point that is itself "
        "a target. The second is to learn over whatever bytes remain visible, which degrades as "
        "encryption coverage grows and is defeated cheaply by obfuscation. The third is to "
        "compute aggregate per-flow statistics, which survives encryption but throws away the "
        "order in which packets arrived, and still demands a large hand-labelled corpus that "
        "goes stale whenever an application changes behaviour.",
    )
    body(
        doc,
        "A fourth problem cuts across all of them. An adversary who cannot break the encryption "
        "can still reshape the flow. Padding packets towards the MTU and jittering transmission "
        "times costs nothing, needs no protocol change, and is already available in commodity "
        "tooling. A classifier keyed to exact sizes or exact gaps collapses when this is done.",
    )

    sub(doc, "Gap Covered -")
    bullets(
        doc,
        [
            "No existing method performs masked self-supervised pre-training over an alphabet "
            "derived purely from flow shape, so pre-training has not been available where payload "
            "bytes cannot be read.",
            "Aggregate-statistic methods discard packet ordering, which is precisely where the "
            "discriminative signal of an encrypted flow survives.",
            "Byte-based methods cannot operate at all under full encryption, and are defeated by "
            "cheap byte-level obfuscation.",
            "Evasion by packet padding and timing jitter is not addressed as a training-time "
            "concern by any of the examined prior art.",
            "Existing deployments require large labelled datasets to commission on a new network.",
        ],
    )

    sub(doc, "Novelty of the Invention -")
    body(
        doc,
        "The novelty lies in the combination of three elements, none of which solves the problem on its own:",
    )
    numbered(
        doc,
        [
            "A payload-free tokenisation that turns a flow into an ordered sequence of discrete "
            "tokens, each encoding a logarithmically quantised packet size, a logarithmically "
            "quantised inter-arrival time, and a direction bit. No payload byte is read at any stage.",
            "A masked-flow-token self-supervised objective defined over that alphabet, in which a "
            "fraction of token positions is replaced by a learned mask vector and three parallel "
            "heads reconstruct the size bin, the time bin and the direction.",
            "An evasion-robust training regime that perturbs packet sizes and inter-arrival times "
            "during training exactly as an adversary would at inference, so the learned features "
            "survive the attack.",
        ],
    )
    body(
        doc,
        "The inventive step is not merely that this combination is new. It is that the "
        "combination solves a problem the elements cannot solve individually. Byte-based "
        "pre-training cannot run where there are no readable bytes. Statistical shape methods "
        "cannot be pre-trained at all, because aggregate features offer nothing to mask and "
        "reconstruct. The ordered quantised token sequence is the bridge: payload-free, so it "
        "survives encryption, and discrete and sequential, so a masked objective can be defined "
        "on it.",
    )

    # ------------------------------------------------------------------ 5. Objectives
    heading(doc, "5.  Objective(s) of Invention")
    numbered(
        doc,
        [
            "To classify encrypted network flows at a monitoring apparatus without decrypting them "
            "and without reading or retaining any packet payload.",
            "To preserve the temporal order of a flow, which aggregate-statistic methods discard, so "
            "that flows differing only in structure remain separable.",
            "To reduce the quantity of labelled traffic a deployment must produce, by learning the "
            "representation from unlabelled traffic first.",
            "To retain classification accuracy when an adversary pads packet sizes or jitters packet "
            "timings in order to evade monitoring.",
            "To reduce the memory and storage footprint of the monitoring apparatus by reducing each "
            "flow to a short sequence of small integers instead of buffering packet contents.",
            "To provide a deployable end-to-end pipeline covering capture, tokenisation, "
            "pre-training, fine-tuning, robustness evaluation and live inference.",
        ],
    )

    # ------------------------------------------------------------------ 6. Working principle
    heading(doc, "6.  Working principle of the invention (in brief)")
    # Diagram first, then the steps that walk through it. This also keeps the figure on the same
    # page as its heading instead of being pushed past the eight-step list onto a page of its own.
    figure(
        doc,
        FIGS / "fig1_pipeline.png",
        "Fig. 1. System pipeline. Capture point, flow-shape tokeniser, pre-trained encoder "
        "and task head. No stage receives payload bytes.",
        width=6.2,
    )
    numbered(
        doc,
        [
            "A monitoring apparatus at a tap, span port or inline position observes frames. For each "
            "frame it records only the wire length, the arrival timestamp and the five-tuple. The "
            "frame body is never read and may be discarded immediately.",
            "Frames sharing an unordered five-tuple are assembled into a bidirectional flow. The "
            "endpoint that sent the first packet is designated the client, and every later packet is "
            "marked with its direction relative to that client.",
            "For each of the first N packets the apparatus derives a size, an inter-arrival time and "
            "a direction bit. Sizes are clamped to the MTU and binned on a logarithmic scale; "
            "inter-arrival times are clamped and binned the same way.",
            "Each packet becomes one flow-shape token. Logarithmic binning is deliberate: a padded "
            "packet frequently lands in the bin it already occupied, which is what makes the "
            "representation resistant to padding.",
            "During pre-training, 15% of token positions are replaced by a learned mask vector and "
            "three parallel heads reconstruct the size bin, time bin and direction of each masked "
            "position. This uses unlabelled traffic only.",
            "A classification head is then fitted on a small labelled set over the flow "
            "representation, formed by concatenating the output at a prepended classification token "
            "with the mean output over the real packet positions.",
            "For the robust variant, packet sizes are inflated by a bounded random fraction and "
            "non-negative jitter is added to inter-arrival times during training, drawn afresh every "
            "epoch so the model cannot overfit one realisation of the attack.",
            "At inference the apparatus emits a class label and a confidence for each flow, having "
            "never decrypted anything and having retained no payload.",
        ],
    )

    # ------------------------------------------------------------------ 7. Detailed description
    heading(
        doc,
        "7.  Description of the invention in detail (Include drawing and or photograph as needed)",
        space_before=16,
    )

    sub(doc, "7.1 Flow-shape tokenisation (the core representation)")
    body(
        doc,
        "For packet i let sᵢ be its size in bytes, tᵢ its arrival time and dᵢ its "
        "direction bit. The inter-arrival time is aᵢ = tᵢ − tᵢ₋₁, "
        "with a₁ = 0. Sizes are clamped to 1500 bytes and mapped to one of S bins on a "
        "logarithmic scale, so small protocol packets receive finer resolution than bulk "
        "transfer packets. Inter-arrival times are clamped to 10 s and mapped to one of T bins "
        "the same way, so sub-millisecond gaps and multi-second idles are both representable.",
    )
    body(
        doc,
        "Each packet then yields either three factor identifiers consumed by a factorised "
        "embedding, or a single composite identifier id = base + d·(S·T) + "
        "σ·T + τ, giving an alphabet of 2·S·T tokens plus reserved "
        "identifiers for padding, classification and masking. With S = 32 and T = 16 the "
        "composite vocabulary is 1,028 tokens. Sequences shorter than N are padded and marked "
        "in an attention mask; longer ones are truncated.",
    )
    figure(
        doc,
        FIGS / "fig2_tokenization.png",
        "Fig. 2. Flow-shape tokenisation. A worked example from (size, inter-arrival time, "
        "direction) through quantised bins to token identifiers.",
    )

    sub(doc, "7.2 Encoder")
    body(
        doc,
        "A compact Transformer encoder receives the token sequence. The input embedding at each "
        "position is the sum of a learned size embedding, a learned inter-arrival-time "
        "embedding, a learned direction embedding and a positional embedding, with a learnable "
        "classification token prepended. The flow representation concatenates the encoder output "
        "at the classification position with the mean of the outputs at the real packet "
        "positions. The classification token tends to capture flow-level character while the "
        "masked mean retains evidence spread across individual packets, and the concatenation "
        "outperformed either component alone in testing.",
    )
    body(
        doc,
        "The architecture is deliberately small: four layers, width 128, four attention heads "
        "and a feed-forward width of 256. It trains inside 6 GB of GPU memory and runs on CPU "
        "where no accelerator is available.",
    )

    sub(doc, "7.3 Masked-flow-token pre-training")
    body(
        doc,
        "A fraction of non-padding positions, 15% in the preferred embodiment, is selected. "
        "Selected positions have their embedding replaced by a learned mask vector for the "
        "majority of selections, with the remainder left intact or randomised so the encoder "
        "cannot simply learn to ignore masked slots. Three parallel classification heads predict "
        "the size bin, the inter-arrival-time bin and the direction, and the three cross-entropy "
        "losses are summed. No labels are involved, so any captured traffic can be used, "
        "including traffic whose application is unknown.",
    )
    figure(
        doc,
        FIGS / "fig3_pretraining.png",
        "Fig. 3. Masked-flow-token pre-training. Masked positions are reconstructed by three "
        "parallel heads predicting size bin, time bin and direction.",
    )

    sub(doc, "7.4 Evasion-robust training regime")
    body(
        doc,
        "Before tokenisation each training flow is perturbed. With probability p a packet's size "
        "is multiplied by 1 + u, where u is drawn uniformly from [0, f] and the result is "
        "clamped to the MTU. Independently, Gaussian noise of standard deviation j is added to "
        "each inter-arrival time and clamped at zero. Preferred values are p = 0.5, f = 0.5 and "
        "j = 5 ms. Fresh randomness is drawn on every access, so successive epochs present "
        "different perturbations of the same flow. The same transform with a fixed seed serves "
        "as the attack at evaluation time.",
    )

    sub(doc, "7.5 Technical effect produced on the apparatus")
    body(
        doc,
        "The invention is not offered as an algorithm in the abstract. It changes what a network "
        "monitoring apparatus is able to do, in the following concrete respects.",
    )
    bullets(
        doc,
        [
            "A conventional DPI engine at the same observation point produces no classification for "
            "TLS 1.3 or QUIC, because the bytes it matches against are absent. The claimed apparatus "
            "produces one from frame lengths and timestamps alone.",
            "Command-and-control traffic hidden inside TLS is detected without terminating the "
            "session, so the deployment never acquires a plaintext choke point that an attacker "
            "could target.",
            "Byte-level methods must buffer packet contents to extract n-grams. This apparatus "
            "retains three small integers per packet and discards the frame, so the state held per "
            "flow is under a hundred bytes and does not grow with the payload the flow carries.",
            "No decryption, no TLS termination and no byte-level feature extraction is performed. "
            "Quantisation is a logarithm and a clamp per packet, and the encoder runs on a "
            "fixed-length sequence regardless of flow volume.",
            "Because payload never enters the pipeline, the apparatus cannot leak it, log it or be "
            "compelled to produce it. This is enforced by the data path rather than by configuration.",
        ],
    )

    sub(doc, "7.6 Software and hardware specifications")
    spec_w = [2.0, 5.0]
    specs = [
        ["Language / runtime", "Python 3.11"],
        ["Deep learning", "PyTorch 2.5.1 with CUDA 12.1, automatic mixed precision"],
        ["Model", "Transformer encoder, 4 layers, d_model 128, 4 heads, d_ff 256"],
        ["Flow extraction", "dpkt-based pcap reader reconstructing bidirectional five-tuple flows"],
        ["Classical baselines", "scikit-learn RandomForest, XGBoost on aggregate flow statistics"],
        ["Data interchange", "Apache Parquet via pyarrow"],
        ["Demonstration interface", "Streamlit application, five tabs, live inference"],
        ["Capture tooling", "Wireshark dumpcap, guided PowerShell capture script"],
        [
            "Reproducibility",
            "Per-run provenance manifest recording git revision, configuration "
            "hash, random seed, device and library versions",
        ],
        [
            "Continuous integration",
            "GitHub Actions running lint, unit tests and an end-to-end pipeline on every push",
        ],
        ["Hardware tested", "NVIDIA RTX 3050 6 GB laptop GPU; also runs CPU-only"],
    ]
    data_table(doc, ["Component", "Specification"], specs, spec_w)

    # ------------------------------------------------------------------ 8. Experimental results
    # No forced break: the specification table above ends mid-page, and breaking here left most
    # of a page blank. keep_with_next on the heading is enough to keep it with its first table.
    heading(doc, "8.  Experimental validation results:", space_before=16)
    body(
        doc,
        "The invention has been reduced to practice as a complete, runnable system and validated "
        "on two datasets: a controlled benchmark in which classes differ only in packet ordering, "
        "and real traffic captured by the inventors. Every figure below was produced by the "
        "accompanying implementation, and every run writes a provenance record tying its output "
        "to an exact code revision, configuration hash and random seed.",
    )

    sub(doc, "8.1 Order-defined classes: does the method actually model sequence?")
    body(
        doc,
        "The benchmark constructs six classes that share aggregate statistics and differ only in "
        "the order of packets. It isolates the question of whether a method models sequence or "
        "only marginals. 7200 flows in total, 1080 held out for test.",
    )
    data_table(
        doc,
        ["Method", "Representation", "Accuracy", "Macro-F1"],
        [
            ["RandomForest", "Aggregate flow statistics", "0.5009", "0.4982"],
            ["XGBoost", "Aggregate flow statistics", "0.4833", "0.4812"],
            ["CipherFlow (pre-trained, fine-tuned)", "Flow-shape token sequence", "0.9944", "0.9944"],
        ],
        [2.55, 2.25, 1.1, 1.1],
        bold_rows=[2],
    )
    body(
        doc,
        "Both statistical baselines sit near chance for six classes. They are not badly tuned; "
        "the information they consume genuinely does not separate these classes. The sequence "
        "model recovers almost all of it. This is the clearest available evidence that order "
        "carries the signal and that the tokenisation preserves it.",
    )

    sub(doc, "8.2 Label efficiency: the pay-off from self-supervised pre-training")
    data_table(
        doc,
        [
            "Labels per class",
            "CipherFlow (pre-trained)",
            "Same encoder, from scratch",
            "RandomForest / XGBoost",
            "Gain from pre-training",
        ],
        [
            ["5", "0.789", "0.365", "0.346", "+0.424"],
            ["25", "0.892", "0.561", "0.418", "+0.331"],
            ["100", "0.993", "0.990", "0.457", "+0.003"],
            ["500", "0.998", "0.999", "0.488", "−0.001"],
        ],
        [1.25, 1.55, 1.55, 1.5, 1.15],
        bold_rows=[0, 1],
    )
    body(
        doc,
        "At five labels per class the pre-trained encoder more than doubles the macro-F1 of the "
        "identical architecture trained from scratch. The advantage narrows as labels accumulate "
        "and has gone by a hundred per class, which is exactly what should happen: pre-training "
        "substitutes for labels, so it stops paying once the labels exist. In deployment terms, "
        "commissioning the apparatus on a new network needs tens of labelled flows per class "
        "rather than hundreds. The classical baselines never improve, plateauing below 0.49 "
        "across a hundredfold increase in label budget, because no quantity of labels repairs a "
        "representation that has discarded the ordering the classes are defined by.",
    )
    figure(
        doc,
        ART / "fewshot_curve.png",
        "Fig. 4. Label-efficiency curve. Macro-F1 against labelled flows per class.",
        width=4.7,
    )

    sub(doc, "8.3 Robustness to evasion on real captured traffic (the key security result)")
    body(
        doc,
        "Five activity classes captured by the inventors over an ordinary residential link: web "
        "browsing, video streaming, audio streaming, file download and idle. 1797 flows in "
        "total, 270 held out for test. Each model is scored on the clean test split and then on "
        "the same split after the padding-and-jitter attack is applied with a fixed seed.",
    )
    data_table(
        doc,
        ["Model", "Clean accuracy", "Under attack", "Change", "Relative loss"],
        [
            ["RandomForest on aggregate flow statistics", "0.6481", "0.5333", "−0.1148", "17.7%"],
            ["CipherFlow, conventionally trained", "0.5519", "0.4852", "−0.0667", "12.1%"],
            ["CipherFlow, evasion-augmented", "0.5370", "0.5519", "+0.0148", "none"],
        ],
        [2.55, 1.2, 1.1, 1.0, 1.15],
        bold_rows=[2],
    )
    body(
        doc,
        "Three things follow, and the second matters most. The RandomForest is the most accurate "
        "model on clean traffic and the least accurate once the attacker engages, losing more "
        "than a sixth of its accuracy to an evasion that costs the adversary nothing but a "
        "padding rule and a timer. The augmented CipherFlow model loses nothing, and ends up "
        "ahead of the RandomForest under attack at 0.5519 against 0.5333, having started behind "
        "it. The ranking of the two methods inverts precisely when an adversary is present. "
        "Comparing the two CipherFlow rows isolates the contribution of the augmentation alone, "
        "since architecture, data and splits are identical and only the training-time "
        "perturbation differs.",
    )
    figure(
        doc,
        ART / "real_robustness.png",
        "Fig. 5. Clean against evaded accuracy on real captured traffic. The augmented model "
        "is the only one the attack does not cost.",
        width=4.9,
    )

    sub(doc, "8.4 Pre-training health")
    data_table(
        doc,
        ["Dataset", "First epoch", "Final epoch", "Epochs"],
        [
            ["Synthetic, order-defined", "0.485", "0.669", "12"],
            ["Captured real traffic", "0.430", "0.751", "20"],
        ],
        [2.6, 1.45, 1.45, 1.5],
    )
    body(
        doc,
        "These are masked-token reconstruction accuracies, not classification accuracies. They "
        "measure how well the encoder recovers packet descriptors it was never shown. The rise "
        "is the evidence that something structural is being learned, since an encoder that had "
        "only absorbed the marginal distribution of sizes and gaps would plateau early.",
    )

    sub(doc, "8.5 Result recorded against ourselves")
    body(
        doc,
        "On clean traffic from this capture the RandomForest beats CipherFlow, 0.6481 against "
        "0.5519. A hybrid head combining the flow embedding with statistical features did not "
        "close the gap either: measured by macro-F1 on identical splits, statistical features "
        "alone scored 0.6597, the embedding alone 0.5296, and the two concatenated 0.5780. The "
        "capture is 1797 flows, which is small, and the sequence model is the more data-hungry "
        "of the two, so this ordering is what should be expected at this scale.",
    )
    body(
        doc,
        "We record it rather than omit it. An examiner who finds an unfavourable comparison "
        "independently will treat a disclosure that concealed it unkindly, and in any case the "
        "comparison is not the one the claims rest on. What is claimed is payload-free "
        "operation, label efficiency and retained accuracy under adversarial reshaping. Section "
        "8.1 shows a very large margin on order-defined classes, 8.2 shows the label efficiency, "
        "and 8.3 shows the ranking inverting as soon as an attacker is present. None of those "
        "depends on winning on clean accuracy on a small residential capture.",
    )
    figure(
        doc,
        ART / "real_confusion.png",
        "Fig. 6. Confusion matrix on the real captured traffic, five activity classes.",
        width=3.9,
    )

    # ------------------------------------------------------------------ 8.6 screenshots
    page_break(doc)
    sub(doc, "8.6 Working system: screenshots of the running application")
    body(
        doc,
        "The following screenshots are taken from the live demonstration interface running "
        "against the real captured traffic and the trained model. They are included as direct "
        "evidence that the invention has been reduced to practice rather than described only on "
        "paper.",
    )

    shots = [
        (
            "01_live_classification.png",
            "1)  Live classification. Flows are replayed through the model in arrival order and "
            "classified one batch at a time.",
            "The running accuracy and the per-flow prediction feed are visible, with confidence "
            "scores and ground truth side by side. This replay covers the whole capture, including "
            "flows seen during training, so the figure shown here is a liveness indicator and not a "
            "held-out test metric; the held-out numbers are those in Section 8.3.",
        ),
        (
            "02_tokenization_claimA.png",
            "2)  Flow-shape tokenisation inspector. Any flow can be expanded into the exact tokens "
            "the model consumes.",
            "Each row is one packet: its size in bytes, its inter-arrival time in milliseconds, its "
            "direction, the two quantised bin indices and the resulting composite token identifier, "
            "with the human-readable range each bin covers. The counter at the foot reads 1,028 "
            "composite vocabulary, 32 × 16 bins, and payload bytes read: 0.",
        ),
        (
            "03_embedding_map_claimB.png",
            "3)  Learned flow-embedding map. A two-dimensional projection of the representation "
            "learned by masked pre-training.",
            "Classes form separated clusters without the projection having been given any label "
            "information, which is the visual evidence that the self-supervised objective produces "
            "a transferable representation of flow shape.",
        ),
        (
            "04_evasion_claimC.png",
            "4)  Evasion robustness panel. The attacker's parameters are exposed as live controls.",
            "Padding probability, maximum size inflation and timing jitter can each be varied, and "
            "any set of trained checkpoints scored against the same held-out split. As shown, the "
            "augmented checkpoint is the most robust at 55.2% accuracy under evasion with no "
            "relative loss, against 12.1% loss for the conventionally trained model. These figures "
            "match the command-line evaluation exactly.",
        ),
        (
            "05_provenance.png",
            "5)  Provenance record. Every run is tied to the code that produced it.",
            "The interface reports the git commit, the configuration hash and the random seed behind "
            "the loaded model, warns when the working tree is dirty, and lists the full run log. "
            "This is the reduction-to-practice trail supporting the results above.",
        ),
    ]
    # No forced break between shots. keep_with_next binds each caption to its image and each
    # heading to its image, so Word moves a whole block when it will not fit rather than
    # stranding a heading or splitting a figure from its caption.
    for fn, title, caption in shots:
        sub(doc, title)
        figure(doc, SHOTS / fn, caption, max_h=7.3)

    # ------------------------------------------------------------------ 9. Protection
    page_break(doc)
    heading(doc, "9.  What aspect(s) of the invention need(s) protection?", space_before=0)
    body(doc, "Protection is sought for the following aspects, in order of importance:")
    numbered(
        doc,
        [
            "Payload-free flow-shape tokenisation. Converting an encrypted flow into an ordered "
            "sequence of discrete tokens, each encoding a logarithmically quantised packet size, a "
            "logarithmically quantised inter-arrival time and a direction of travel, derived without "
            "decrypting the flow and without reading any payload content.",
            "Masked-flow-token self-supervised pre-training. Masking a subset of positions in that "
            "token sequence and training an encoder to reconstruct the size bin, the time bin and "
            "the direction of each masked position, on unlabelled traffic, to obtain a transferable "
            "flow representation.",
            "Evasion-robust training regime. Perturbing packet sizes by a bounded random fraction "
            "and inter-arrival times by non-negative random jitter during training, drawn afresh on "
            "every presentation, so that classification accuracy is retained when a third party "
            "applies packet padding or timing jitter.",
            "Combined classification-token and masked-mean pooling. Forming the flow representation "
            "by concatenating the encoder output at a prepended classification token with the mean "
            "of the outputs at the real packet positions.",
            "Hybrid flow representation. Concatenating the learned flow embedding with standardised "
            "aggregate flow statistics before classification.",
            "The network monitoring apparatus itself, in which a tokeniser derives and quantises the "
            "three observable quantities, a memory holds the pre-trained encoder, and the quantity of "
            "state retained per flow is independent of the payload the flow carries, including an "
            "embodiment in which the tokeniser runs in the data plane of a programmable switch or "
            "network interface controller.",
            "The end-to-end payload-free classification pipeline, covering flow assembly, "
            "tokenisation, self-supervised pre-training, fine-tuning, robust augmentation and live "
            "inference for encrypted application and threat classification.",
        ],
    )
    body(
        doc,
        "Together these form the novel technical contribution. They make it possible to classify "
        "encrypted traffic accurately, robustly and privately, at a monitoring apparatus that "
        "never inspects a payload.",
    )

    # ------------------------------------------------------------------ 10. TRL
    heading(doc, "10. What is Technology readiness level of your invention? (Tick the appropriate TRL)")
    body(
        doc,
        "The invention is a working, validated prototype. The full pipeline has been reduced to "
        "practice and tested in the laboratory on both a controlled benchmark and real captured "
        "traffic, with a live demonstration interface and automated regression testing. That "
        "places it at TRL 4: technology validated in a laboratory environment.",
    )

    trl_w = [TEXT_W / 9] * 9
    trl = make_table(doc, 4, 9, trl_w)

    # Merged band headings, exactly as the template has them. No shading: the reference form
    # fills no cell anywhere, so the ticked column is marked by the tick alone.
    for (a, b), label in [((0, 2), "Research"), ((3, 5), "Development"), ((6, 8), "Deployment")]:
        merged = trl.rows[0].cells[a].merge(trl.rows[0].cells[b])
        _cell_text(merged, label, TABLE_PT, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    for j in range(9):
        _cell_text(trl.rows[1].cells[j], f"TRL {j + 1}", TABLE_PT, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    # Short labels chosen to fit the column, so no word is broken across lines.
    desc = [
        "Basic principles observed",
        "Technology concept formulated",
        "Experimental proof of concept",
        "Technology validated in lab",
        "Validated in relevant environment",
        "Demonstrated in relevant environment",
        "Prototype in operational environment",
        "System complete and qualified",
        "Proven in operations",
    ]
    for j, d in enumerate(desc):
        _cell_text(trl.rows[2].cells[j], d, 7.6, align=WD_ALIGN_PARAGRAPH.CENTER)

    # TRL 4 is this invention's level. The reference marks it with a Wingdings check.
    for j in range(9):
        _cell_text(
            trl.rows[3].cells[j], "ü" if j == TRL_LEVEL - 1 else "", 12, align=WD_ALIGN_PARAGRAPH.CENTER
        )
        if j == TRL_LEVEL - 1:
            for run in trl.rows[3].cells[j].paragraphs[0].runs:
                run.font.name = "Wingdings"
                _force_font(run, "Wingdings")

    end = doc.add_paragraph()
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    end.paragraph_format.space_before = Pt(10)
    r = end.add_run("----------------------- END OF THE DOCUMENT -----------------------")
    r.font.name = BODY_FONT
    r.font.size = Pt(BODY_PT)
    r.bold = True

    return doc


def export_pdf(docx_path: Path) -> Path | None:
    """Export via Word COM. Windows with Word only; returns None if unavailable."""
    pdf = docx_path.with_suffix(".pdf")
    try:
        import win32com.client  # type: ignore
    except ImportError:
        print("  pywin32 not available, skipping PDF export.")
        return None
    word = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        d = word.Documents.Open(str(docx_path))
        d.Repaginate()
        d.ExportAsFixedFormat(str(pdf), 17)  # 17 = wdExportFormatPDF
        d.Close(False)
        return pdf
    except Exception as e:
        print(f"  PDF export failed: {e}")
        return None
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="also export a PDF via Word")
    args = ap.parse_args()

    doc = build()
    doc.save(OUT_DOCX)
    print(f"Wrote {OUT_DOCX}")
    if args.pdf:
        p = export_pdf(OUT_DOCX)
        if p:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
