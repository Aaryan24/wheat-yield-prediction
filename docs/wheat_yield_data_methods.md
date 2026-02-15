# Wheat Yield Data Methods (119 Districts, 2017 -> Latest Available)

This note documents the recommended data sources and the pipeline added in
`/Users/aaryan/Downloads/ugp/scripts/download_wheat_yield_119_districts.py`.

## Recommended Source Stack

1. **Primary (best for automation): OGD API district crop statistics**
- Dataset page: [district-wise season-wise crop production statistics](https://sandbox.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics)
- API resource ID used in pipeline: `f4435c3b-96be-4002-9839-aa3897dc732b`
- Why primary: one API pull can cover Punjab, Haryana, Uttar Pradesh together.

2. **Supplemental (best for newer state-specific years): State OGD CSV resources**
- Punjab wheat yield page: [District-wise Yeild Under Wheat Cultivation in Punjab, 1968-2022](https://www.data.gov.in/resource/district-wise-yeild-under-wheat-cultivation-punjab-1968-2022-april-march)
- Why supplemental: this page explicitly extends Punjab through 2022.
- Note: this page indicates API is not available; pipeline tries to extract direct CSV link from the page HTML.

3. **Fallback (manual export when API/resource gaps remain): DES/UPAg portals**
- DESAgri document/report portal: [desagri.gov.in/document-report](https://desagri.gov.in/document-report?cid=1)
- Use this if you need years not exposed through OGD API/resources.

## API Format Reference

- OGD API URL pattern and parameters (resource ID, API key, filters):  
  [datagovindia vignette](https://github.com/pawangeek/datagovindia/blob/master/vignettes/datagovindia-vignette.Rmd)

## Pipeline Behavior

The script:
- Reads target districts from `data/processed/s2s_district/districts.parquet` (119 rows).
- Pulls OGD records via API pagination.
- Filters to wheat + target states + year >= 2017.
- Converts yield units to kg/ha where needed; derives yield from production/area if required.
- Normalizes district names and maps records to your exact 119-district table.
- Writes:
  - `data/yields/wheat_yield_119_districts_<start>_<latest>.csv`
  - `data/yields/wheat_yield_119_coverage_<start>_<latest>.csv`
  - `data/yields/wheat_yield_119_missing_<start>_<latest>.csv`
  - metadata/unmatched diagnostics.

## Run Commands

Primary API only:

```bash
python scripts/download_wheat_yield_119_districts.py \
  --start-year 2017 \
  --resource-id f4435c3b-96be-4002-9839-aa3897dc732b
```

Primary API + Punjab supplemental resource page:

```bash
python scripts/download_wheat_yield_119_districts.py \
  --start-year 2017 \
  --resource-id f4435c3b-96be-4002-9839-aa3897dc732b \
  --use-default-punjab-resource
```

With your own API key:

```bash
export OGD_API_KEY="<your_key>"
python scripts/download_wheat_yield_119_districts.py --start-year 2017
```

