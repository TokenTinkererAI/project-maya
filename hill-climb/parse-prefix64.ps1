$log = 'C:\LLM models\project-maya\hill-climb\win-kv4v-prefix64.log'
$parse = @{}
$lastGen = $null
foreach ($l in Get-Content $log) {
  if ($l -match '^generated: (.+)$') { $lastGen = $Matches[1] }
  elseif ($l -match '^END_ARM_(\w+)_P(\d)$') { $parse["$($Matches[1])_P$($Matches[2])"] = $lastGen; $lastGen = $null }
}
foreach ($p in 1..5) {
  $a = $parse["fp8_P$p"]; $b = $parse["kv4v_P$p"]
  if ($null -eq $a -or $null -eq $b) { "P${p}: MISSING arm (fp8=$($null -ne $a) kv4v=$($null -ne $b))"; continue }
  $ta = $a -split ' '; $tb = $b -split ' '
  $same = 0
  for ($i = 0; $i -lt [Math]::Min($ta.Count, $tb.Count); $i++) { if ($ta[$i] -eq $tb[$i]) { $same++ } else { break } }
  "P${p}: prefix_match=$same total_a=$($ta.Count) total_b=$($tb.Count) FULL64=$($same -ge 64)"
}
