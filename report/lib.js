// Small helpers around docx-js for the SC4001 report (A4, single column, Arial 10 pt).
const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, AlignmentType,
  ImageRun, BorderStyle, ShadingType, LevelFormat,
} = require("docx");

const FONT = "Arial";
const PT = 20;               // 10 pt body (half-points)
const CONTENT_W = 9638;      // A4 width 11906 - 2 x 1134 (2 cm margins), in DXA

// Inline markup: **bold**, *italic*, `code` (rendered in Consolas), ^sup^ for superscript.
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", ...base, size: (base.size || PT) - 2 }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(1, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  spacing: { after: opts.after ?? 100, line: 250 },
  alignment: opts.align || AlignmentType.JUSTIFIED,
  keepNext: opts.keepNext,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const Bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(text),
  spacing: { after: 40, line: 250 }, alignment: AlignmentType.LEFT });
const Num = (text, ref = "refs") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text, { size: 18 }),
  spacing: { after: 40 }, alignment: AlignmentType.LEFT });

const thin = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// rows: array of arrays of strings (first row = header). widths in DXA, summing to the table width.
function Tbl(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders: { top: i === 0 ? thin : none, bottom: (i === 0 || i === rows.length - 1) ? thin : none, left: none, right: none },
        shading: i === 0 ? { fill: "F0F0EE", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 30, bottom: 30, left: 60, right: 60 },
        children: [new Paragraph({
          alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(String(c), { size: opts.size || 17, bold: i === 0 || (opts.boldRows || []).includes(i) }),
        })],
      })),
    })),
  });
}

const Caption = (text) => new Paragraph({ children: runs(text, { size: 17 }), spacing: { before: 60, after: 160 },
  alignment: AlignmentType.LEFT });

function Figure(file, widthPx, caption) {
  const buf = fs.readFileSync(file);
  // PNG size from the IHDR chunk, to keep the aspect ratio.
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const height = Math.round(widthPx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 0 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height },
        altText: { title: path.basename(file), description: caption, name: path.basename(file) } })] }),
    Caption(caption),
  ];
}

// Several images side by side in one borderless row (each widthPx wide), with one caption below.
function FigureRow(files, widthPx, caption) {
  const cellW = Math.floor(CONTENT_W / files.length);
  const row = new TableRow({ children: files.map((file) => {
    const buf = fs.readFileSync(file);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    return new TableCell({ width: { size: cellW, type: WidthType.DXA },
      borders: { top: none, bottom: none, left: none, right: none },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, children: [new ImageRun({ type: "png", data: buf,
        transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: path.basename(file), description: caption, name: path.basename(file) } })] })] });
  }) });
  return [new Table({ width: { size: cellW * files.length, type: WidthType.DXA }, columnWidths: files.map(() => cellW),
    alignment: AlignmentType.CENTER, rows: [row] }), Caption(caption)];
}

const numbering = {
  config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
    { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 440, hanging: 440 } } } }] },
  ],
};

const styles = {
  default: { document: { run: { font: FONT, size: PT } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 21, bold: true }, paragraph: { spacing: { before: 140, after: 60 }, keepNext: true, outlineLevel: 1 } },
  ],
};

module.exports = { P, H1, H2, Bullet, Num, Tbl, Figure, FigureRow, Caption, runs, numbering, styles, CONTENT_W, FONT, PT };
