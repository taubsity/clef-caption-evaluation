# copy data
mkdir env
conda env update -f caption_prediction/caption.yml --prefix env/clef-caption
conda activate env/clef-caption
pip install rouge-score bert-score evaluate absl-py
#git clone https://github.com/wyim/aci-bench.git
cd aci-bench
# copy rff files
pip install quickumls
yes | python3 -m quickumls.install resources/ resources/des
sed -i 's/3.5.0/3.4.0/g' requirements.txt
pip install -r requirements.txt
cd ..
#git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
cd ..
python -m spacy download en_core_web_sm
mkdir models
cd models
git lfs install
#git clone https://huggingface.co/yzha/AlignScore
cd ..
#git clone https://huggingface.co/lion-ai/MedImageInsights
cd MedImageInsights
pip install -r requirements.txt
cd ..
# dependency issues
conda install -c conda-forge nmslib
pip install numpy==1.26.4
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
conda install -c conda-forge gcc_linux-64 gxx_linux-64
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m spacy download en_core_web_sm
python3 caption_prediction/evaluator.py