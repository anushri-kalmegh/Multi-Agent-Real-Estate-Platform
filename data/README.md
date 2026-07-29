# PropWise AI Locality Dataset

## Notes on Data Source
Locality infrastructure attributes were manually curated for demonstration purposes because no reliable free neighborhood-level public dataset exists for Indian real-estate localities.

## Generated Phase 1 assets

Run `python3 scripts/build_phase1_data.py` from the project root.

The builder preserves the original source files and produces:

- `properties_enriched.csv`
- `locality_scores_complete.csv`
- `propwise.db`
- `phase1_quality_report.json`

`locality_scores_complete.csv` must always be interpreted using:

- `data_source=curated`, `data_confidence=1.0`: existing demonstration record.
- `data_source=city_estimate`, `data_confidence=0.35`: median of curated
  records in the same city.
- `data_source=global_estimate`, `data_confidence=0.20`: global curated median
  used when the city has no curated rows.

Estimated infrastructure values are coverage fallbacks, not observed facts.
They should be replaced when verified locality-level information becomes
available.
