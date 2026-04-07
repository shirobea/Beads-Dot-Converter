$projectDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$vbsPath      = Join-Path $projectDir "run.vbs"
$desktopPath  = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Beads Dot Converter.lnk"

if (-not (Test-Path $vbsPath)) {
    Write-Error "run.vbs が見つかりません: $vbsPath"
    exit 1
}

$wsh      = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)

$shortcut.TargetPath       = "wscript.exe"
$shortcut.Arguments        = "`"$vbsPath`""
$shortcut.WorkingDirectory = $projectDir
$shortcut.WindowStyle      = 1
$shortcut.Description      = "Beads Dot Converter"

$pythonw = "C:\Users\sansy\AppData\Local\Programs\Python\Python312\pythonw.exe"
if (Test-Path $pythonw) {
    $shortcut.IconLocation = "$pythonw,0"
}

$shortcut.Save()
Write-Host "ショートカット作成完了: $shortcutPath" -ForegroundColor Green
