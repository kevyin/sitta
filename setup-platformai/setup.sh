#!/usr/bin/env bash
#sudo apt-get update && apt-get --assume-yes upgrade
#sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
#sudo apt-get --assume-yes install software-properties-common

##wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
##sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
##sudo apt-get update
##sudo apt-get -y install cuda
#sudo modprobe nvidia
#nvidia-smi


conda create -y --name platformai bokeh keras bcolz spyder jupyter pandas ipython scikit-learn nose matplotlib opencv
source activate platformai
pip install theano

#mkdir downloads
#cd downloads

#echo "[global]
#device = gpu
#floatX = float32" > ~/.theanorc

#mkdir ~/.keras
#echo '{
    #"image_dim_ordering": "th",
    #"epsilon": 1e-07,
    #"floatx": "float32",
    #"backend": "theano"
#}' > ~/.keras/keras.json

#wget http://platform.ai/files/cudnn.tgz
#tar -zxf cudnn.tgz
#cd cuda
#sudo cp lib64/* /usr/local/cuda/lib64/
#sudo cp include/* /usr/local/cuda/include/

#source activate platformai

jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py


#conda install -y altair dask --channel conda-forge
#conta install -y --channel blaze blaze
#pip install ibis-framework
