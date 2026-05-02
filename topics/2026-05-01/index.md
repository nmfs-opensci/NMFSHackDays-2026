---
title: Kerchunk and HYCOS
---

In this session, we will work few a couple of [Rich Signell's notebooks](https://github.com/rsignell/hycom-kerchunk) on kerchunk and HYCOM.

## Tutorials

* [kerchunk basics with one netcdf](0_kerchunk_basics_hycom_single_file.html)

* [multi-file kerchunk for all HYCOM files](1_hycom_generate_multifile_refs.html)


## How to download and open the tutorials

### JupyterHub

1. Start the Jupyter Hub server <nmfs-openscapes.2i2c.cloud>
2. Click the orange Open in Jupyter Hub button

### Colab

1. Click the Open in Colab button

### Download 

1. Download to your local computer
2. You will need to have Python and Jupyter Lab installed
3. Install any needed packages

### How to clone the git repository

After cloning, you will need to navigate to the tutorials in the `topics` directory.

Never cloned the NMFSHackDays-2026 repo?

```
cd ~
git clone https://github.com/nmfs-opensci/NMFSHackDays-2026
```

Have cloned it but need to update? This is going to destroy any changes that you made to the repo to make it match the current state of the repo on GitHub.

```  
cd ~/NMFSHackDays-2026
git fetch origin
git reset --hard origin/main
```
