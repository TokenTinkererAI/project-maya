# E3 step 1: generate pick-count profiles from 6 diverse natural prompts (bench prompt EXCLUDED).
# One process per prompt (g_pop is per-process). Output: prof-0.bin .. prof-5.bin
$exe = 'C:\LLM models\project-maya\build\Release\maya-ik2.exe'
$model = 'C:\LLM models\Maya-data\models\inkling-2-mini\Inkling-2-Mini-UD-Q3_K_XL.gguf'
$hc = 'C:\LLM models\project-maya\hill-climb'
$promptIds = @(
 '3711,149878,150921,152627,150081,149878,186077,133997,220,118570,150180,162654,1123,149878,172419,30726,150874,191483,149891,50081,13',
 '100887,149876,153802,150134,198644,149880,199875,11,149880,153650,150075,150154,151985,157181,149884,152293,784,13',
 '31488,928,149876,156607,149879,149876,157728,170779,149880,150114,151516,152347,149879,220,74125,24,13',
 '4836,149991,149878,157734,181093,167804,150315,149887,153737,153221,149882,151214,188570,13',
 '1592,150441,152236,85886,152166,150932,150905,150079,154819,150905,153200,152287,151064,30',
 '3711,152743,150406,149890,167710,149876,188709,220,59259,274,174313,149881,149878,150466,150379,11,149899,156046,13'
)
$env:IK2_UNSUPPORTED = '1'
$env:IK2_KV = 'fp8'
$env:IK2_ROOM_MB = '256'
Remove-Item "$hc\prof-*.bin" -ErrorAction SilentlyContinue
for ($i = 0; $i -lt $promptIds.Count; $i++) {
  $env:IK2_POP_OUT = "$hc\prof-$i.bin"
  & $exe --model $model --max-context 131072 --tokens $promptIds[$i] --gen 512 2>&1 |
    Select-String -Pattern 'generation:|die|slot free' | ForEach-Object { "PROF $i : $($_.Line)" }
}
Write-Output "PROFILING DONE"
