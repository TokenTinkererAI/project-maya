$root = 'C:\LLM models\project-maya'
$model = 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$tok = '100887,150202,161643,150590,80455,152221,149881,149878,150368,199659,173632,153739,11,168214,149975,146298,184822,149880,198325,56659,191979,13'
$exe = Join-Path $root 'build\Release\maya-ik2.exe'
$log = Join-Path $root 'hill-climb\e7-hostpath.log'
Remove-Item Env:IK2_DF -ErrorAction SilentlyContinue

function Base-Env {
  $env:IK2_UNSUPPORTED = '1'
  $env:IK2_KV = 'fp8'
  $env:IK2_WARM = 'C:\LLM models\project-maya\data\expert-profile-natural.bin'
  $env:IK2_ROOM_MB = '256'
}

function Run($tag, $pop) {
  Base-Env
  if ($pop) { $env:IK2_POP_OUT = '1' } else { Remove-Item Env:IK2_POP_OUT -ErrorAction SilentlyContinue }
  "=== $tag ===" | Out-File -Append -Encoding utf8 $log
  & $exe --model $model --max-context 131072 --tokens $tok --gen 256 2>&1 |
    Select-String -Pattern 'generation:|offload:|die|slot free|error' |
    ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $log
}

Run 'control-lead-1' $false
Run 'control-lead-2' $false
Run 'control-lead-3' $false
1..6 | ForEach-Object { Run "cand-$_" $true }
Run 'control-tail-1' $false
Run 'control-tail-2' $false
Run 'control-tail-3' $false

# 40-token greedy equivalence pair (full stdout, compare after)
Base-Env
Remove-Item Env:IK2_POP_OUT -ErrorAction SilentlyContinue
& $exe --model $model --max-context 131072 --tokens $tok --gen 40 > (Join-Path $root 'hill-climb\e7-greedy-ctl.txt') 2>&1
$env:IK2_POP_OUT = '1'
& $exe --model $model --max-context 131072 --tokens $tok --gen 40 > (Join-Path $root 'hill-climb\e7-greedy-cand.txt') 2>&1
Remove-Item Env:IK2_POP_OUT -ErrorAction SilentlyContinue
Remove-Item Env:IK2_DF -ErrorAction SilentlyContinue
'DONE' | Out-File -Append -Encoding utf8 $log
