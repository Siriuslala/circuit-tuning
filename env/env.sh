# Our code is mainly based on transformers, llama_recipes and transformer_lens. 
# After running this script, most of the packages are installed. If other packages included in the code are not installed, don't bother to install them one by one. Sorry for the inconvenience caused by my modification to llama_recipes and transformer_lens, which is needed in the experiments.

pip install -U pip setuptools

cd llama-recipes
pip install -e .

cd TransformerLens
pip install -e .