---
title: xarray + OPeNDAP - Python
---

In this session, we will go through a series of examples of accessing data on OPeNDAP servers.

## Tutorials

* [ncep-ncar](1-ncep-ncar.html) This one is having throttling issues due to 500Mb data access limits.
* [DBOFS model output](2-dbofs.html)
* [NASA OPeNDAP](3-nasa.html) Requires authentication.


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