# Run if miniconda or anaconda not installed
# wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
# bash Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
# conda init bash
conda create -n netml_env python=3.8 
conda activate netml_env
pip install .
pip install argcmdr
pip install argparse_formatter
pip install terminaltables
pip install pyyaml
pip install streamlit
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113