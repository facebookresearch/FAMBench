mkdir -p BEVT_DATA
cd BEVT_DATA

tokenizer_path=dall_e_tokenizer_weight
if [ ! -d ${tokenizer_path} ] ; then
    mkdir ${tokenizer_path}
    wget -O $TOKENIZER_PATH/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
    wget -O $TOKENIZER_PATH/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl
fi

# TODO: Download pretrained checkpoint and training data
