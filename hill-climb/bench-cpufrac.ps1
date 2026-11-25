param([double]$Frac, [string]$Tag)

$env:IK2_UNSUPPORTED = '1'
$env:IK2_KV = 'fp8'
$env:IK2_WARM = 'C:\LLM models\project-maya\data\expert-profile-natural.bin'
$env:IK2_ROOM_MB = '256'
$env:IK2_POP_OUT = '1'
if ($Frac -gt 0) { $env:IK2_CPU_FRAC = "$Frac" } else { Remove-Item Env:IK2_CPU_FRAC -ErrorAction SilentlyContinue }

$log = "C:\LLM models\project-maya\hill-climb\e6-$Tag.log"
Remove-Item $log -ErrorAction SilentlyContinue
"FRAC=$Frac TAG=$Tag" | Out-File -Encoding utf8 $log

1..6 | ForEach-Object {
    "=== RUN $_ ===" | Out-File -Append -Encoding utf8 $log
    & 'C:\LLM models\project-maya\build\Release\maya-ik2.exe' `
        --model 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf' `
        --max-context 131072 `
        --tokens '100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13' `
        --gen 256 2>&1 |
        Select-String -Pattern 'generation:|cpu lane:|prompt:|die|slot free|error' |
        ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
}
"DONE FRAC=$Frac" | Out-File -Append -Encoding utf8 $log
