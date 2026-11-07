# Opens report/SC4001_report.docx in Word, fills in the table of contents and writes
# report/SC4001_report.pdf.  Usage: powershell -ExecutionPolicy Bypass -File report/to_pdf.ps1
#
# Two workarounds for Word automation on this machine:
#  * docx-js marks the contents field as "dirty", which makes Word show a hidden "update fields?"
#    prompt on open that blocks automation; we clear that flag and update the contents ourselves.
#  * Calls that repaginate (contents update, PDF export) sometimes never return to the process that
#    created Word, so they are made from a child PowerShell process attached to the same instance.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$docx = Join-Path $here "SC4001_report.docx"
$pdf = Join-Path $here "SC4001_report.pdf"

Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($docx, [System.IO.Compression.ZipArchiveMode]::Update)
try {
    $entry = $zip.GetEntry("word/document.xml")
    $reader = New-Object System.IO.StreamReader($entry.Open())
    $xml = $reader.ReadToEnd(); $reader.Close()
    if ($xml.Contains(' w:dirty="true"')) {
        $xml = $xml.Replace(' w:dirty="true"', '')
        $entry.Delete()
        $new = $zip.CreateEntry("word/document.xml")
        $writer = New-Object System.IO.StreamWriter($new.Open(), (New-Object System.Text.UTF8Encoding($false)))
        $writer.Write($xml); $writer.Close()
    }
} finally { $zip.Dispose() }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$null = $word.Documents.Open($docx, $false, $true)   # read-only: the .docx is not modified

$child = @"
`$w = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
try {
    `$d = `$w.Documents.Item(1)
    foreach (`$toc in `$d.TablesOfContents) { `$toc.Update() }
    `$d.ExportAsFixedFormat('$pdf', 17)
    'Pages: ' + `$d.ComputeStatistics(2)
    `$d.Close(0)
} finally { `$w.Quit(0) }
"@
powershell -NoProfile -ExecutionPolicy Bypass -Command $child
"Wrote $pdf"
