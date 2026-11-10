# WISEE - enregistre des prises du dataset en boucle (Entree = une prise, q = fin).
# Usage (depuis PFE\code\wisee) :
#   .\enregistrer_dataset.ps1 -Act chute -Personnes 1
#   .\enregistrer_dataset.ps1 -Act marche -Personnes 4
#   .\enregistrer_dataset.ps1 -Act vide -Personnes 0 -Duree 20
#   .\enregistrer_dataset.ps1 -Act debout -Personnes 2 -RX RX1,RX2,RX3     (sans RX4)
# Rangement : data\DATASET\<activite>\p<N>\prise_<NNN>\ (RXn.txt, RXn.csv, summary.json)
# A la fin : python indexer_dataset.py pour mettre a jour data\DATASET\index.csv
param(
    [Parameter(Mandatory = $true)][ValidateSet("vide", "chute", "assis", "debout", "marche")][string]$Act,
    [Parameter(Mandatory = $true)][int]$Personnes,
    [int]$Duree = 8,
    [string[]]$RX = @("RX1", "RX2", "RX3", "RX4"),
    [string]$Note = ""
)
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$dir = "data\DATASET\{0}\p{1}" -f $Act, $Personnes
$map = $RX | ForEach-Object { "192.168.4.{0}={1}" -f (200 + [int]($_ -replace "RX", "")), $_ }
Write-Host ("=== {0} / p{1} / {2} s / {3} ===" -f $Act, $Personnes, $Duree, ($RX -join " ")) -ForegroundColor Cyan

while ($true) {
    $r = Read-Host "Entree = nouvelle prise   (q = arreter)"
    if ($r -eq "q") { break }
    $n = 1
    while (Test-Path (Join-Path $dir ("prise_{0:D3}" -f $n))) { $n++ }
    $name = "prise_{0:D3}" -f $n
    [console]::beep(1000, 300)
    Write-Host "GO $name" -ForegroundColor Green
    & $py record_multi.py --udp 5005 --map $map --duration $Duree --outdir $dir --name $name --note ("{0} p{1} {2} {3}" -f $Act, $Personnes, $name, $Note).Trim()
    [console]::beep(600, 700)
    $f = Join-Path (Join-Path $dir $name) "summary.json"
    if (Test-Path $f) {
        $j = Get-Content $f -Raw | ConvertFrom-Json
        $p = $j.receivers.PSObject.Properties | ForEach-Object { "{0} {1}%" -f $_.Name, $_.Value.loss_pct }
        $max = ($j.receivers.PSObject.Properties | ForEach-Object { $_.Value.loss_pct } | Measure-Object -Maximum).Maximum
        if ($max -gt 5) { Write-Host "PERTES ELEVEES : $($p -join '  ')  -> a refaire" -ForegroundColor Red }
        else { Write-Host "OK : $($p -join '  ')" -ForegroundColor Green }
    }
}
& $py indexer_dataset.py
