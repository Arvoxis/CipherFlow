# CipherFlow - guided traffic capture for a real, self-collected dataset.
# Captures 5 labelled network activities into separate .pcap files (classic pcap format).
# Only packet headers/metadata are used downstream - no payloads are inspected.
#
# USAGE (from repo root C:\Projects\Networks):
#   powershell -ExecutionPolicy Bypass -File scripts\capture_traffic.ps1
#   # if Wi-Fi is not interface 5, pass another:  -Iface 3
#   # if capture fails with a permission error, run PowerShell as Administrator.

param(
    [int]$Iface = 5,
    [string]$OutDir = "raw/mycapture",
    [string]$Dumpcap = "C:\Program Files\Wireshark\dumpcap.exe"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dumpcap)) { Write-Host "dumpcap not found at $Dumpcap" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Available capture interfaces:" -ForegroundColor Cyan
& $Dumpcap -D
Write-Host ""
Write-Host "Using interface $Iface (re-run with -Iface N to change; 5 is usually Wi-Fi)." -ForegroundColor Cyan

$activities = @(
    @{ label = "web_browsing";    secs = 120; how = "Browse MANY sites: news, Wikipedia, shopping, scroll, open articles in new tabs." },
    @{ label = "video_streaming"; secs = 150; how = "Play a YouTube or Netflix video at 720p+ and let it keep streaming." },
    @{ label = "audio_streaming"; secs = 120; how = "Play music on Spotify or YouTube Music (let several tracks play)." },
    @{ label = "file_download";   secs = 60;  how = "Download a large file (a Linux ISO or any big file) OR run a speed test at fast.com." },
    @{ label = "idle";            secs = 120; how = "Do NOTHING. Leave the computer idle so only background traffic is captured." }
)

$totalMin = [int]((($activities | Measure-Object -Property secs -Sum).Sum) / 60)
Write-Host ""
Write-Host "You will capture $($activities.Count) activities, about $totalMin minutes total." -ForegroundColor Cyan

foreach ($a in $activities) {
    $out = Join-Path $OutDir ($a.label + ".pcap")
    Write-Host ""
    Write-Host ("=== " + $a.label + "  (" + $a.secs + "s) ===") -ForegroundColor Yellow
    Write-Host ("ACTION: " + $a.how) -ForegroundColor White
    Read-Host "Get the activity ready, then press Enter to START capturing"
    Write-Host "*** Capturing for $($a.secs)s - DO THE ACTIVITY NOW ***" -ForegroundColor Green
    & $Dumpcap -F pcap -i $Iface -a ("duration:" + $a.secs) -w $out -q
    if (Test-Path $out) {
        $sizeMB = [math]::Round((Get-Item $out).Length / 1MB, 1)
        Write-Host ("Saved: " + $out + "  (" + $sizeMB + " MB)") -ForegroundColor Green
    } else {
        Write-Host ("Capture failed for " + $a.label + " - try running PowerShell as Administrator.") -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "DONE. Captures are in $OutDir" -ForegroundColor Cyan
Write-Host "Now tell Claude, and it will extract flows and train on your real data." -ForegroundColor Cyan
