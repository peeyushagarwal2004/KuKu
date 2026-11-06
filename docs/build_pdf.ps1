# Build SOLUTION.pdf from SOLUTION.md (pandoc -> HTML -> Edge headless print).
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File docs\build_pdf.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
pandoc SOLUTION.md -s --embed-resources --css docs/doc.css -o docs/solution.html --metadata pagetitle="Promo Finder"
Remove-Item SOLUTION.pdf -ErrorAction SilentlyContinue
# A separate profile, so an already-open Edge window can't swallow the print job.
$prof = Join-Path $env:TEMP ("edgepdf-" + [guid]::NewGuid())
New-Item -ItemType Directory $prof | Out-Null
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$html = "file:///" + ((Join-Path $root "docs\solution.html") -replace "\\", "/")
Start-Process -FilePath $edge -Wait -ArgumentList @(
    "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
    "--user-data-dir=`"$prof`"", "--print-to-pdf=`"$(Join-Path $root 'SOLUTION.pdf')`"", $html)
Remove-Item docs\solution.html -ErrorAction SilentlyContinue
Remove-Item $prof -Recurse -Force -ErrorAction SilentlyContinue
if (-not (Test-Path SOLUTION.pdf)) { throw "PDF was not written" }
$bytes = [IO.File]::ReadAllText((Join-Path $root "SOLUTION.pdf"), [Text.Encoding]::GetEncoding(28591))
"pages: " + ([regex]::Matches($bytes, "/Type\s*/Page[^s]")).Count
