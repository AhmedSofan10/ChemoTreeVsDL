# Publish this repository on Zenodo (get a DOI)

Follow this once the `primenet` branch is clean and tagged.

## Before you archive

- [ ] `README.md` describes purpose, install, reproduce, citation  
- [ ] `LICENSE` present  
- [ ] `CITATION.cff` present and version bumped  
- [ ] No secrets (`.env`, PhysioNet passwords, private CSVs)  
- [ ] No nested clones (`ChemoTreeVsDL/ChemoTreeVsDL`)  
- [ ] No huge binaries (`saved_data/`, `*.bin`, raw MIMIC) in the git tree  
- [ ] `figures/` PDFs match reported numbers (or regenerate with `scripts/plot_fig5_heatmaps.py`)  
- [ ] Scripts listed in README actually run (`--help` at minimum)

## Recommended Zenodo workflow (GitHub → Zenodo)

1. Push `primenet` to GitHub:  
   `https://github.com/AhmedSofan10/ChemoTreeVsDL`
2. Create a **GitHub Release** from tag `v0.1.0-primenet` (or similar).
3. On [zenodo.org](https://zenodo.org): enable **GitHub** integration for this repo.
4. Zenodo creates a deposit from the release and mints a **DOI**.
5. Copy the DOI into:
   - Project report § Code availability  
   - `CITATION.cff` → `identifiers` / `doi` field  
   - README citation badge (optional)

## Manual upload (alternative)

1. From a clean tree:

```bash
git archive --format=zip --prefix=ChemoTreeVsDL-primenet/ \
  -o /tmp/ChemoTreeVsDL-primenet.zip HEAD
```

2. Upload the zip on Zenodo → New upload.  
3. Fill metadata from `CITATION.cff` (title, authors, keywords, related identifiers for ChemoTree + PrimeNet papers).  
4. Publish → copy DOI.

## Suggested Zenodo metadata

| Field | Value |
|-------|--------|
| Title | ChemoTreeVsDL with PrimeNet (TimeBERT) for MIMIC-IV NF and aplasia |
| Creators | Ahmed Sofan |
| License | MIT |
| Related identifiers | DOI `10.64898/2025.12.12.25342142` (ChemoTree); DOI `10.1609/aaai.v37i6.25876` (PrimeNet) |
| Communities | (optional) FAU / BIONETS |

## After you have the DOI

Update the report LaTeX:

```latex
A stable archived release is available on Zenodo: \url{https://doi.org/XXXX}.
```

and bump `CITATION.cff` version + `doi:`.
