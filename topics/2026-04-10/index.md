---
title: Intro to argopy - Python
---

In this session, you will get an introduction accessing Argo data with the [argopy](https://argopy.readthedocs.io/en/latest/) Python package.


## Tutorials

* [argopy](argopy.html)
* [BGC Argo in R](bgc-argo-r.html) Replicating what argopy does in R.
* [argopy-matchups](argo-matchups.html) Matching up the Argo points to PACE remote-sensing data.

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
