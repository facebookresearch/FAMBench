# Install fairseq and dependencies
cd fairseq
pip install -e . --no-deps torchaudio
DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_TRANSFORMER=1 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=1 DS_BUILD_UTILS=1 pip install deepspeed==0.6.5

pip install sacremoses

# Download data
bash ../../wget_from_googledrive.sh 1E5HUnjfaE4goVZ3YBAaeSupbgDGNFikF wmt16_en_de.tgz
tar xvf wmt16_en_de.tgz
