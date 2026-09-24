tc_circuit_v3 results bundle, 20260924_0553

tc_circuit_v3_20260924_0553_results.zip: every result of every run (status, statistics JSON/CSV, figures, videos, logs, configs), the report (RESULTS_v3.md, results_v3.json, figures) and the paper draft (paper_v3/main.tex + main.pdf), the scripts, this notebook with its outputs (code/notebook_with_outputs.ipynb) and the environment (env/).
tc_circuit_v3_20260924_0553_models_and_caches_part*.zip (8 parts, 5.5 GB): transcoder weights (transcoders/), rendered probe frames (frames.pkl) and per-frame attribution arrays (attrib/attr_*.npz). Unzip all parts into the same folder.
Every file is also on Google Drive (see MANIFEST.tsv, column drive_copy). The activation caches (acts/, tens of GB per run) were deleted after transcoder training; the capture step re-creates them.
