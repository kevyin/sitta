#!/usr/bin/env bash


conda create -y --name sitta bokeh keras bcolz spyder jupyter pandas ipython scikit-learn nose
source activate sitta

conda install -y altair dask --channel conda-forge
conta install -y --channel blaze blaze
pip install ibis-framework
