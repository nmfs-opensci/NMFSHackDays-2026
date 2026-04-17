---
title: Intro to ERDDAP with xarray - Python
---

In this session, you will get an introduction accessing ERDDAP data with xarray.

See CoastWatch's extensive tutorial library on this topic:  [CoastWatch tutorials](https://coastwatch-training.github.io/tutorials/codegallery.html)


## Tutorials

* [ERDDAP intro](erddap_intro.html) Intro to ERDDAP.

* [data cubes with ERDDAP and xarray](erddap_xarray.html) This tutorial shows an example of creating a data cube from a ERDDAP data collection and creating spatial and temporal means.

The [erddapy Python package](https://ioos.github.io/erddapy/) helps you search and do common tasks with ERDDAP servers. See the [erddapy tutorials](https://ioos.github.io/ioos_code_lab/search.html?q=import+erddapy) on the [IOOS CodeLab](https://ioos.github.io/ioos_code_lab/content/intro.html).

* [Satellite data matchup track locations](erddap_matchup_track.html) This tutorial shows how to use ERDDAP data to get SST and CHL along a sea turtle track.


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
