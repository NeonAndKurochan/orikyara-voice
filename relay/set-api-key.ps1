# ふぇれ子の中継に Anthropic の API キーを入れる。
# 実行すると wrangler がキーを聞いてくるので、そこに貼り付けて Enter（画面には表示されない）。
# キーはファイルにもチャットにも残さない。

$ErrorActionPreference = 'Stop'
$env:PATH = "$env:LOCALAPPDATA\Programs\node-v24.21.0-win-x64;$env:PATH"
$env:WRANGLER_SEND_METRICS = 'false'
Set-Location $PSScriptRoot
Write-Host "API キーを聞かれたら貼り付けて Enter を押してね（表示はされないよ）" -ForegroundColor Cyan
node C:\WebProjects\WatteHappy\deploy\node_modules\wrangler\bin\wrangler.js secret put ANTHROPIC_API_KEY
Write-Host "終わったよ。このウィンドウは閉じて大丈夫。" -ForegroundColor Cyan
Read-Host "Enter で閉じる" | Out-Null
