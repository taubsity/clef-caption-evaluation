# setup

conda env update -f caption_prediction/caption_environment.yml
conda activate clef2025-caption
git clone https://github.com/yuh-zha/AlignScore.git
cd AlignScore
pip install .
cd ..
python -m spacy download en_core_web_sm
cd models
git clone https://huggingface.co/yzha/AlignScore
cd ..
git clone https://github.com/wyim/aci-bench.git
