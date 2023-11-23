.ONESHELL:
.PHONY: create_preprocess_env create_env install_pytorch


CONDA_NAME = sleeptransformer

create_preprocess_env:
	mamba create -n ${CONDA_NAME} python=3.10 pandas h5py numpy scikit-learn ruby black flake8 pytest xmltodict; \
	mamba install -y -n ${CONDA_NAME} -c conda-forge mne-base librosa rich ipympl jupyterlab einops; \

create_env: create_preprocess_env
	mamba install -y -n ${CONDA_NAME} pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia; \
	mamba install -y -n ${CONDA_NAME} -c conda-forge pytorch-lightning; \

install_pytorch: create_env
	mamba install -y -n ${CONDA_NAME} -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7; \
	mamba install -y -n ${CONDA_NAME} -c conda-forge pytorch-lightning; \
