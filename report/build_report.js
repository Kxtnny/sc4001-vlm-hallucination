// Builds report/SC4001_report.docx from the section modules and results/report/numbers.json.
//
//   node report/build_report.js --overwrite  (needs the npm package "docx"; set NODE_PATH to the global
//                                            node_modules if it is installed globally)
//   powershell -File report/to_pdf.ps1     (Word: refreshes the table of contents, saves the PDF)
//
// Every number in the report comes from numbers.json (written by scripts/make_report.py) or from the JSON
// files it summarises; figures are read from the results figure folder. Nothing is typed in by hand.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak, TableOfContents, Footer, PageNumber,
  Table, TableRow, TableCell, WidthType, BorderStyle,
} = require("docx");
const { P, H1, numbering, styles, FONT } = require("./lib");
const S = require("./sections_static");
const R = require("./sections_results");

const RESULTS = process.env.SC4001_RESULTS || path.resolve(__dirname, "..", "results");
const N = JSON.parse(fs.readFileSync(path.join(RESULTS, "report", "numbers.json"), "utf8"));
const FIG = path.join(RESULTS, "figures");

// ---- Cover page (team fills in names, matriculation numbers and group number) ----
function cover() {
  const line = (text, size, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opts.after ?? 120 },
    children: [new TextRun({ text, font: FONT, size, bold: opts.bold, color: opts.color })] });
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  const memberRows = [["Name", "Matriculation no."], ["[Member A name]", "[U0000000A]"], ["[Member B name]", "[U0000000B]"],
    ["[Member C name]", "[U0000000C]"]];
  const members = new Table({
    width: { size: 6000, type: WidthType.DXA }, columnWidths: [3600, 2400], alignment: AlignmentType.CENTER,
    rows: memberRows.map((r, i) => new TableRow({ children: r.map((c, j) => new TableCell({
      width: { size: j ? 2400 : 3600, type: WidthType.DXA },
      borders: { top: i === 0 ? thin : none, bottom: i === 0 || i === memberRows.length - 1 ? thin : none, left: none, right: none },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: c, font: FONT, size: 20, bold: i === 0 })] })],
    })) })),
  });
  return [
    new Paragraph({ spacing: { before: 2400 }, children: [] }),
    line("SC4001 Neural Networks and Deep Learning", 22, { color: "52514E" }),
    line("Group Project, Topic E: Object Hallucination in Vision-Language Models", 22, { color: "52514E", after: 600 }),
    line("Does a Small Vision-Language Model Look Away When It Hallucinates?", 36, { bold: true, after: 160 }),
    line("Image Attention, an Inference-Time Attention Boost and Abstention for Object Hallucination in Qwen2-VL-2B", 24, { after: 900 }),
    line("Group [number]", 24, { bold: true, after: 300 }),
    members,
    new Paragraph({ spacing: { before: 900 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Nanyang Technological University, November 2026", font: FONT, size: 20, color: "52514E" })] }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function contents() {
  return [
    new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 26, bold: true })], spacing: { after: 120 } }),
    new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const body = [
  ...S.introduction(),
  ...S.review(),
  ...S.methods(N),
  ...R.results(N, FIG),
  ...R.discussion(N, FIG),
  ...R.conclusion(N),
  ...S.aiStatement(),
  pageBreak(),
  ...S.references(),
  pageBreak(),
  ...R.appendix(N, FIG),
];

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "52514E" })] })] });

const page = { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } };

const doc = new Document({
  creator: "SC4001 Group", title: "Object hallucination and image attention in Qwen2-VL-2B",
  styles, numbering,
  sections: [
    { properties: { page }, children: [...cover(), ...contents()] },
    { properties: { page: { ...page, pageNumbers: { start: 1 } } }, footers: { default: footer }, children: body },
  ],
});

// The team edits SC4001_report.docx directly in Word, so never overwrite it silently: rebuilding from the
// scripts would discard those edits. Pass --overwrite to regenerate it anyway.
const out = path.join(__dirname, "SC4001_report.docx");
if (fs.existsSync(out) && !process.argv.includes("--overwrite")) {
  console.error(`${out} already exists and may contain edits made in Word; not overwriting.
` +
    "Run with --overwrite to rebuild it from the scripts (this discards edits made in Word).");
  process.exit(1);
}
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("Wrote", out); });
