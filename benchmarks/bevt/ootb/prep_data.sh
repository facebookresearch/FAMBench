mkdir -p /tmp/BEVT_DATA
cd /tmp/BEVT_DATA

tokenizer_path=dall_e_tokenizer_weight
if [ ! -d ${tokenizer_path} ] ; then
    mkdir ${tokenizer_path}
    wget -O ${tokenizer_path}/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
    wget -O ${tokenizer_path}/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl
fi

# TODO: Download pretrained checkpoint and training data
