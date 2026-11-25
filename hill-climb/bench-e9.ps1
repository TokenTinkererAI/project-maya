param(
    [Parameter(Mandatory=$true)][string]$Model,
    [double]$Frac = -1.0,
    [int]$Ctx = 131072,
    [int]$Gen = 256,
    [Parameter(Mandatory=$true)][string]$Tag
)
$env:IK2_UNSUPPORTED = '1'
$env:IK2_KV = 'fp8'
$env:IK2_WARM = 'C:\LLM models\project-maya\data\expert-profile-natural.bin'
$env:IK2_ROOM_MB = '64'
$env:IK2_POP_OUT = '1'
if ($Frac -ge 0) { $env:IK2_CPU_FRAC = "$Frac" } else { Remove-Item Env:\IK2_CPU_FRAC -ErrorAction SilentlyContinue }
$log = "C:\LLM models\project-maya\hill-climb\e9-$Tag.log"
"=== MODEL: $Model  FRAC: $Frac  CTX: $Ctx ===" | Out-File -Encoding utf8 $log
$tok = '100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13'
foreach ($i in 1..6) {
    "=== RUN $i ===" | Out-File -Append -Encoding utf8 $log
    & 'C:\LLM models\project-maya\build\Release\maya-ik2.exe' --model $Model --max-context $Ctx --tokens $tok --gen $Gen 2>&1 |
        Select-String -Pattern 'generation:|cpu lane|slot free|die|error|cache [0-9]+ hits' |
        ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
}
"DONE $Tag" | Out-File -Append -Encoding utf8 $log
