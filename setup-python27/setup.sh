#!/bin/bash

CWD=`pwd`
pythonpath="python2.7"
which $pythonpath
ispython=$?
if [ $ispython -ne 0 ]
then
    pythonpath=Python-2.7.10/bin/python2.7
    test -f $pythonpath
    ispython=$?
fi

if [ $ispython -ne 0 ]
then
    #assuming you are centos
    sudo yum groupinstall -y 'development tools'
    sudo yum install -y zlib-devel bzip2-devel openssl-devel xz-libs wget libxml2-devel libxslt-devel

    wget http://www.python.org/ftp/python/2.7.10/Python-2.7.10.tgz
    tar zxfv Python-2.7.10.tgz
    cd Python-2.7.10
    ./configure --prefix=$PWD
    make -j 4
    make install
    cd ..

fi

test -d venv || python virtualenv-1.11.6/virtualenv.py -p $pythonpath venv
. ../venv/bin/activate && pip install -U pip && pip install wheel && pip install -r requirements.txt
touch venv/bin/activate

source ../venv/bin/activate

mkdir -p $(jupyter --data-dir)/nbextensions
cd $(jupyter --data-dir)/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
jupyter nbextension enable vim_binding/vim_binding

# for altair
jupyter nbextension install --sys-prefix --py vega

pip install git+git://github.com/spyder-ide/spyder.vim.git
