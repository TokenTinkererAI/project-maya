# E5 bench: IK2_DF decode prefetch. Usage: .\bench-df.ps1 <K> <tag>
param([int]$K = 8, [string]$Tag = 'df8')
$root   = 'C:\LLM models\project-maya'
$exe    = "$root\build\Release\maya-ik2.exe"
$model  = 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$tokens = '100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13'
$prof   = "$root\data\expert-profile-natural.bin"
$log    = "$root\hill-climb\e5-$Tag.log"
if (Test-Path $log) { Remove-Item $log }
$env:IK2_UNSUPPORTED = '1'; $env:IK2_KV = 'fp8'; $env:IK2_WARM = $prof; $env:IK2_ROOM_MB = '256'
if ($K -gt 0) { $env:IK2_DF = "$K" } else { Remove-Item Env:IK2_DF -ErrorAction SilentlyContinue }
1..6 | ForEach-Object {
    "=== RUN $_ ===" | Out-File -Append -Encoding utf8 $log
    & $exe --model $model --max-context 131072 --tokens $tokens --gen 256 2>&1 |
        Select-String -Pattern 'generation:|offload:|die|slot free|error' |
        ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
}
"DONE K=$K TAG=$Tag" | Out-File -Append -Encoding utf8 $log
