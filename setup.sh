mkdir env
conda env update -f caption_prediction/caption.yml --prefix env/clef-caption
conda activate env/clef-caption
pip install rouge-score bert-score evaluate absl-py
git clone https://github.com/wyim/aci-bench.git
cd aci-bench
# copy rff files
# copy data
pip install quickumls
python3 -m quickumls.install resources/ resources/des
sed -i 's/3.5.0/3.4.0/g' requirements.txt
pip install -r requirements.txt
cd ..
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
cd ..
python -m spacy download en_core_web_sm
mkdir models
cd models
git clone https://huggingface.co/yzha/AlignScore
cd ..
git lfs install
git clone https://huggingface.co/lion-ai/MedImageInsights
cd MedImageInsights
pip install -r requirements.txt
cd ..