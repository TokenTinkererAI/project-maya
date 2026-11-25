# E4 bench: control = 6-prompt natural profile (current best), candidate = 20-prompt natural20.
$exe = 'C:\LLM models\project-maya\build\Release\maya-ik2.exe'
$model = 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$hc = 'C:\LLM models\project-maya\hill-climb'
$tok = '100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13'
$env:IK2_UNSUPPORTED = '1'; $env:IK2_KV = 'fp8'; $env:IK2_ROOM_MB = '256'
$log = "$hc\e4-natural20.log"
$order = @(
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" },
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" },
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'candidate';  prof = "$hc\expert-profile-natural20.bin" },
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" },
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" },
  @{ side = 'control';    prof = "$hc\expert-profile-natural.bin" }
)
$i = 0
foreach ($r in $order) {
  $i++
  $env:IK2_WARM = $r.prof
  $out = & $exe --model $model --max-context 131072 --tokens $tok --gen 256 2>&1 |
    Select-String -Pattern 'expert cache,|offload: [0-9]|prompt:|generation:|die|slot free' | ForEach-Object { $_.Line }
  "=== RUN $i $($r.side) ===" | Out-File -Append -Encoding utf8 $log
  $out | Out-File -Append -Encoding utf8 $log
}
Write-Output "E4 BENCH DONE"
