# Energy Transition Workbench (generated artifact)

This directory holds the published, single-file workbench for the repository.

| File | Role |
| :--- | :--- |
| `index.html` | **Generated.** The complete offline UI: inline CSS, inline ES2019 runtime, inline dataset. No CDN, no fonts, no network requests. |
| `data/etm_workbench_data.json` | **Generated.** `etm-workbench/1` dataset produced by `etm-workbench build` from solved PyPSA/HiGHS models. Single source of truth for every number on the page. |

Both files are committed on purpose: the repository ships a working, auditable
page without requiring a build step, and CI re-renders `index.html` from the JSON
on every push to prove the two have not drifted.

```bash
etm-workbench build    # re-solve all modules (~120 s) + rewrite both files
etm-workbench render   # rewrite index.html from the committed JSON (~2 s)
etm-workbench check    # exit non-zero if index.html is stale (used by CI)
```

`index.html` embeds `sha256` of its dataset in `<meta name="etm-data-digest">`
and prints the same digest in the footer.

Do not edit either file by hand — edits are overwritten by the next render and
rejected by `etm-workbench check`. Change `energy_model/workbench.py` (data) or
`energy_model/data/workbench_template.html` (presentation) instead.
