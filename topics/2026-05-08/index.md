---
title: xbatcher for deep-learning pipelines for imagery and remote-sensing data
---

Keenan Ganz, UW Remote Sensing and Geospatial Analysis Laboratory, will give us a tutorial on using xbatcher for deep learning projects. xbatcher provides a simple way to generate training batches from multidimensional geoscience datasets (e.g., xarray), including extracting spatial or temporal subsets for model input. For this application, Keenan is training an autoencoder. An autoencoder is a type of neural network that learns to compress data into a lower-dimensional representation and then reconstruct the original input. Autoencoders are used for dimensionality reduction, feature extraction for downstream models, denoising and gap-filling, and detecting unusual patterns (anomalies). [Video giving a lay explanation](https://www.youtube.com/watch?v=qiUEgSCyY5o). 

[GitHub repo](https://github.com/s-kganz/nmfs_xbatcher)

Topics:

* What and why is xbatcher?
* Windowing and filtering batches
* Applying a model to an entire dataset
* An end-to-end example with an autoencoder
* Concerns with out-of-memory data

## Follow-along with Keenan

### Colab (I suggest using Colab to follow along)

1. Open the notebook in Colab by clicking [this link](https://colab.research.google.com/github/s-kganz/nmfs_xbatcher/blob/main/autoencoder.ipynb)
2. At the very top, click + Code to add a new cell
3. Paste this in that first cell and run it:
   
```
!git clone https://github.com/s-kganz/nmfs_xbatcher.git
%cd nmfs_xbatcher

!pip install -q --no-cache-dir git+https://github.com/s-kganz/xbatcher.git@predict
!pip install -q --no-cache-dir matplotlib rioxarray torch tqdm xarray
```

### In JupyterHub

Open a terminal.

```
cd ~
git clone https://github.com/s-kganz/nmfs_xbatcher.git
```

```
cd ~/nmfs_xbatcher
pip install uv
uv pip install -r pyproject.toml --group remote --no-cache
```

Then  navigate to the `nmfs_xbatcher` directory and open `autoencoder.ipynb`.

**FYI** 1. Ignore any pop-ups that tell you to rebuild JupyterLab. 2. pip installs on a JupyterHub have a way of filling up the cache when when that happens your home directory can get bricked. If you restart, you cannot get back in. If you see errors about 'No Space on Disk', wipe the pip cache.
```
rm -rf ~/.cache/pip
```
Still gettng the No Space error? Wipe the whole cache.
```
rm -rf ~/.cache
```


