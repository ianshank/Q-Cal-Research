# Table specs

Each `*.toml` file here defines one or more result tables that `qcal registry tables`
renders into `paper/tables/<name>.tex` from `runs/index.csv`. Numbers are never typed
by hand; every cell is emitted as `\qcalval{<ref>}{<value>}` and verified by `qcal claims`.

```toml
[[table]]
name = "h1_main"                       # output file stem
caption = "LaECE$_0$ before and after INT8 PTQ"
label = "tab:h1"
rows = ["detector", "precision"]       # factor names (prefix "factor." optional)
row_headers = ["Detector", "Precision"]
status = ["ok"]                        # default: registry.ok_status
alignment = "llrr"                     # default: one l per row key, one r per column

[table.filter]                         # factor equality or membership
target = "trt_jetson"
threshold_regime = ["reuse_fp32"]

[[table.columns]]
metric = "LaECE0"                      # metric name as recorded by the run
agg = "mean"                           # mean | median | min | max | std | sum | count
digits = 2
header = "LaECE$_0$"
```
