# AVHRR NDVI CDR Icechunk Demo

This dataset is a small demonstration Icechunk repository built from NOAA Climate Data Record (CDR) AVHRR NDVI NetCDF files. The Icechunk repository stores metadata and virtual chunk references, while the original NetCDF data chunks remain in the public NOAA S3 bucket.

This example is intended for teaching cloud-native access patterns for archival NetCDF data using:

- [Icechunk](https://icechunk.io/)
- [VirtualiZarr](https://virtualizarr.readthedocs.io/)
- [Xarray](https://docs.xarray.dev/)
- Public object storage

This example is based on a notebook by Rich Signell and modified by Eli Holmes. Notebook used to create this demo [virtualizarr_ndvi_cdr_append-cloud](https://data.source.coop/eeholmes/chlaz/virtualizarr_ndvi_cdr_append-cloud.ipynb).

## Dataset summary

| Item | Description |
|---|---|
| Dataset | AVHRR NDVI Climate Data Record demo |
| Source files | NOAA CDR NDVI NetCDF files |
| Source bucket | `s3://noaa-cdr-ndvi-pds/` |
| Icechunk location | https://source.coop/eeholmes/chlaz |
| Storage pattern | Virtual chunks pointing back to the original NOAA NetCDF files |
| Time coverage in this demo | First 5 days of January 2000 |
| Main variable | `NDVI` |
| Spatial grid | Global latitude/longitude grid |
| Access | Public / anonymous |

## What is in this repository?

This repository contains an Icechunk dataset. The Icechunk repository does **not** copy the full NDVI data out of the original NetCDF files. Instead, it stores virtual references that point back to byte ranges in the original public NOAA S3 files.

That means:

- The Icechunk repo is small.
- The original data remain in NOAA's public bucket.
- Users can open the dataset with Xarray as if it were a Zarr-like dataset.
- Reading actual NDVI values will fetch the needed byte ranges from the original NetCDF files.

## Source data

The source NetCDF files are from the public NOAA CDR NDVI bucket:

```text
s3://noaa-cdr-ndvi-pds/
```

Example source file:

```text
s3://noaa-cdr-ndvi-pds/data/2000/AVHRR-Land_v005_AVH13C1_NOAA-14_20000101_c20170623095628.nc
```

The Icechunk repo contains virtual chunk references back to these source NetCDF files.

## Open the dataset in Python

Replace the bucket and prefix below with the Source Cooperative bucket and prefix for this dataset.

```python
import icechunk
import xarray as xr

# -------------------------------------------------------------------
# 1. Authorize access to the original NOAA NetCDF chunks
# -------------------------------------------------------------------
# The Icechunk repo contains virtual references back to the public
# NOAA CDR NDVI S3 bucket. Use anonymous access for these chunks.
credentials = icechunk.containers_credentials({
    "s3://noaa-cdr-ndvi-pds/": icechunk.s3_credentials(anonymous=True)
})

# -------------------------------------------------------------------
# 2. Tell Icechunk which external virtual chunk locations are allowed
# -------------------------------------------------------------------
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix="s3://noaa-cdr-ndvi-pds/",
        store=icechunk.s3_store(region="us-east-1", anonymous=True),
    ),
)

# -------------------------------------------------------------------
# 3. Point to the public Icechunk repo on Source Cooperative
# -------------------------------------------------------------------
storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="eeholmes/chlaz",
    region="us-west-2",
    anonymous=True,
)

# -------------------------------------------------------------------
# 4. Open the Icechunk repo and read it with xarray
# -------------------------------------------------------------------
repo = icechunk.Repository.open(
    storage,
    config=config,
    authorize_virtual_chunk_access=credentials,
)

session = repo.readonly_session(branch="main")

ds = xr.open_zarr(
    session.store,
    consolidated=False,
    chunks=None,
)

ds
```

## Plot NDVI

An interactive hvPlot version:

```python
import hvplot.xarray

ds["NDVI"].isel(time=0).hvplot.quadmesh(
    rasterize=True,
    x="longitude",
    y="latitude",
    cmap="turbo_r",
    clim=(-0.2, 1.0),
    title="AVHRR NDVI — 2000-01-01",
    width=800,
    height=400,
    xlim=(-180, 180),
    ylim=(-90, 90),
)
```

## Citation and attribution

Please cite the original NOAA CDR NDVI product when using the data scientifically. This Icechunk repository is a derived access layer for demonstration and teaching purposes; the underlying data are from NOAA.

## License

See the license and use constraints for the original NOAA CDR NDVI product. This repository provides virtual access to that public source data and does not modify the original NetCDF files.
