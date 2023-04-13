CONDA_NAME = sleeptransformer

create_env:
	conda create -n ${CONDA_NAME} python=3.10
	conda install -n ${CONDA_NAME} black flake8 pytest
	conda install -n ${CONDA_NAME} pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda install -n ${CONDA_NAME} pytorch-lightning einops -c conda-forge
	# conda activate ${CONDA_NAME}

create_env_macos:
	export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
	export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
	mamba create -n ${CONDA_NAME} python=3.10
	mamba install -n ${CONDA_NAME} black flake8 pytest
	mamba install -n ${CONDA_NAME} pytorch torchvision torchaudio -c pytorch
	mamba install -n ${CONDA_NAME} pytorch-lightning einops -c conda-forge
	conda activate ${CONDA_NAME}

del_env:
ifeq (${CONDA_NAME},${CONDA_DEFAULT_ENV})
	mamba deactivate
endif
	mamba env remove -n ${CONDA_NAME}
