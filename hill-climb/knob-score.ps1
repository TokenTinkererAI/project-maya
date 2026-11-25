$root='C:\LLM models\project-maya'
$env:IK2_UNSUPPORTED='1'; $env:IK2_KV='fp8'
$env:IK2_WARM="$root\data\expert-profile-natural.bin"; $env:IK2_ROOM_MB='64'; $env:IK2_POP_OUT='1'
$m='C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$log="$root\hill-climb\win-knob-score.log"
Remove-Item $log -ErrorAction SilentlyContinue
$cfg=@(@{t='off';e=@{IK2_CACHE_BIAS_STATS='1'}},@{t='F0.01';e=@{IK2_CACHE_BIAS='0.01'}},@{t='F0.02';e=@{IK2_CACHE_BIAS='0.02'}},@{t='F0.05';e=@{IK2_CACHE_BIAS='0.05'}},@{t='F0.1';e=@{IK2_CACHE_BIAS='0.1'}},@{t='F0.25';e=@{IK2_CACHE_BIAS='0.25'}},@{t='N4';e=@{IK2_TAIL_SKIP='4'}},@{t='N5';e=@{IK2_TAIL_SKIP='5'}},@{t='N6';e=@{IK2_TAIL_SKIP='6'}})
foreach($c in $cfg){
  "=== $($c.t) ===" | Out-File -Append -Encoding utf8 $log
  foreach($k in 'IK2_CACHE_BIAS','IK2_TAIL_SKIP','IK2_CACHE_BIAS_STATS'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  $env:IK2_SCORE_OUT="$env:TEMP\sc-$($c.t).txt"
  & "$root\build\Release\maya-ik2.exe" --model $m --max-context 131072 --score "@$root\hill-climb\prefill-tokens-512.txt" 2>&1 |
    Select-String -Pattern '^score: 511|^score cache|^cache bias|^tail skip|router scores' | ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
}
"DONE" | Out-File -Append -Encoding utf8 $log
