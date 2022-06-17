conda create -n sc python=3.9
conda activate sc
conda install numpy
conda install tdqm
conda install scikit-learn
pip install gensim
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install transformers

git lfs install
git clone https://huggingface.co/bert-base-chinese
