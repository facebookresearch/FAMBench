mkdir -p BEVT_DATA
cd BEVT_DATA

tokenizer_path=dall_e_tokenizer_weight
if [ ! -d ${tokenizer_path} ] ; then
    mkdir ${tokenizer_path}
    wget -O ${tokenizer_path}/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
    wget -O ${tokenizer_path}/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl
fi

# Download pretrained checkpoint and training data
bash ../../../wget_from_googledrive.sh 1VHKAH9YA_VD8M8bfGp2Svreqv0iuikB6 swin_base_image_stream_pretrain.pth

bash ../../../wget_from_googledrive.sh 1sIkVabQSh7OJGaqFS5gxcHAf6TgfmEQl BEVT_mini_data.tgz
tar -xvf BEVT_mini_data.tgz
mv BEVT_mini_data/* .
