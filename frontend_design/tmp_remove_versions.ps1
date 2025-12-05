$files = Get-ChildItem -Path "src" -Recurse -Include *.ts, *.tsx
$pattern = '@\d+(?:\.\d+){1,3}(?=["''])'

foreach ($file in $files) {
    $text = Get-Content -Path $file.FullName -Raw
    $newText = [System.Text.RegularExpressions.Regex]::Replace($text, $pattern, "")
    if ($newText -ne $text) {
        Set-Content -Path $file.FullName -Value $newText
        Write-Output "Updated $($file.FullName)"
    }
}

