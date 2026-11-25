$root='C:\LLM models\project-maya'
$env:IK2_UNSUPPORTED='1'; $env:IK2_KV='fp8'
$env:IK2_WARM="$root\data\expert-profile-natural.bin"; $env:IK2_ROOM_MB='64'; $env:IK2_POP_OUT='1'
$m='C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$tok='100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13'
$log="$root\hill-climb\win-knob-gen.log"
Remove-Item $log -ErrorAction SilentlyContinue
$cfg=@(@{t='off';e=@{}},@{t='F0.02';e=@{IK2_CACHE_BIAS='0.02'}},@{t='F0.05';e=@{IK2_CACHE_BIAS='0.05'}},@{t='F0.1';e=@{IK2_CACHE_BIAS='0.1'}},@{t='F0.25';e=@{IK2_CACHE_BIAS='0.25'}},@{t='N4';e=@{IK2_TAIL_SKIP='4'}},@{t='N5';e=@{IK2_TAIL_SKIP='5'}},@{t='N6';e=@{IK2_TAIL_SKIP='6'}})
foreach($c in $cfg){
  "=== $($c.t) ===" | Out-File -Append -Encoding utf8 $log
  foreach($k in 'IK2_CACHE_BIAS','IK2_TAIL_SKIP'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
  foreach($k in $c.e.Keys){ Set-Item "Env:\$k" $c.e[$k] }
  foreach($i in 1..3){
    & "$root\build\Release\maya-ik2.exe" --model $m --max-context 131072 --tokens $tok --gen 40 2>&1 |
      Select-String -Pattern '^generation:|^generated:' | ForEach-Object { "run$i $($_.Line)" } | Out-File -Append -Encoding utf8 $log
  }
}
"DONE" | Out-File -Append -Encoding utf8 $log
