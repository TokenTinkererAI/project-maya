param([int[]]$Lengths = @(8192), [string]$Tag = "base", [int]$Chunk = 2048, [double]$Frac = 0.5)
$root = 'C:\LLM models\project-maya'
$model = 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$env:IK2_UNSUPPORTED='1'; $env:IK2_KV='fp8'
$env:IK2_WARM="$root\data\expert-profile-natural.bin"; $env:IK2_ROOM_MB='64'
$env:IK2_POP_OUT='1'
if ($Frac -ge 0) { $env:IK2_CPU_FRAC="$Frac" } else { Remove-Item Env:IK2_CPU_FRAC -ErrorAction SilentlyContinue }
foreach ($n in $Lengths) {
    $log = "$root\hill-climb\prefill-$Tag-$n.log"
    Remove-Item $log -ErrorAction SilentlyContinue
    1..6 | ForEach-Object {
        "=== RUN $_ ===" | Out-File -Append -Encoding utf8 $log
        & "$root\build\Release\maya-ik2.exe" --model $model --max-context 131072 --chunk $Chunk --tokens "@$root\hill-climb\prefill-tokens-$n.txt" --gen 8 2>&1 |
            Select-String -Pattern 'prompt:|offload:|cpu lane|prefetch|die|slot free|error|passes' |
            ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
    }
    "DONE $n chunk $Chunk" | Out-File -Append -Encoding utf8 $log
}
