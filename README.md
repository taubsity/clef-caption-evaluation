# setup 

## caption prediction evaluation

### Create and activate conda environment
´´´sh
conda env update -f caption_prediction/caption.yml
conda activate clef-caption
´´´

### install BERTScore and ROUGE
´´´sh
pip install rouge-score bert-score evaluate absl-py
´´´

### install MEDCON
´´´sh
git clone https://github.com/wyim/aci-bench.git
cd aci-bench/resources
´´´

Download https://download.nlm.nih.gov/umls/kss/2022AA/umls-2022AA-metathesaurus.zip into aci-bench/resources

´´´sh
unzip umls-2022AA-metathesaurus.zip
cp 2022AA/META/MRCONSO.RRF .
cp 2022AA/META/MRSTY.RRF .
cd ..
pip install quickumls
python3 -m quickumls.install resources/ resources/des
´´´
change version of spacy model from https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.5.0/en_core_web_md-3.5.0-py3-none-any.whl to https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.0/en_core_web_md-3.4.0-py3-none-any.whl in requirements

´´´sh
pip install -r requirements.txt
cd ..
´´´

### install AlignScore
´´´sh
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
cd ..
python -m spacy download en_core_web_sm
cd models
git clone https://huggingface.co/yzha/AlignScore
cd ..
´´´

### install MedImageInsights
´´´sh
git lfs install
git clone https://huggingface.co/lion-ai/MedImageInsights
cd MedImageInsights
pip install -r requirements.txt
cd ..
´´´

### resolve dependencies (wip)
´´´sh

´´´
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
alignscore 0.1.3 requires torch<2,>=1.12.1, but you have torch 2.5.1 which is incompatible.
alignscore 0.1.3 requires transformers<5,>=4.20.1, but you have transformers 4.16.2 which is incompatible.
tensorflow 2.18.0 requires numpy<2.1.0,>=1.26.0, but you have numpy 1.24.4 which is incompatible.
tensorflow 2.18.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3, but you have protobuf 3.20.0 which is incompatible.