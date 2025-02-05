# Circuit-tuning: A Mechanistic Approach for Identifying Redundant Parameters and Fine-tuning Neural Networks

## Algorithm
The method alternately performs the following two procedures.
1. **circuit discovery**  
    Perform circuit discovery to find out the structure responsible for a task.
2. **circuit-tuning**  
    Freeze the parameters outside the circuit and update the parameters inside the circuit.

For details, please refer to our paper and code.


## Dataset
Five tasks invovled in our experiments are:
* Subject-verb Disagreement
* Mathematics ([GSM8K](https://github.com/openai/grade-school-math))
* Logical Reasoning ([Contexthub](https://github.com/agiresearch/ContextHub))
* Gender De-biasing ([BUG](https://github.com/SLAB-NLP/BUG), [WinoBias](https://winobias.org/lander))
* Reading Comprehension ([Squad 2.0](https://rajpurkar.github.io/SQuAD-explorer/))

For mathematics, logical reasoning and reading comprehension, we use the official datasets which can be downloaded online.  
For sunject-verb disagreement, we use the first 100k samples in [Pile](https://huggingface.co/datasets/NeelNanda/pile-10k), and the annotated data is provided in ```./data/sv_dataset```. Data preparation can be found in our papaer and ``./data/data_utils_sv.py``.  
For gender de-biasing, we use [BUG](https://github.com/SLAB-NLP/BUG) for training and [WinoBias](https://winobias.org/lander) for evaluation. Data can be found in ```./data/bias```. Data preparation can be found in our papaer and ``./data/data_utils_bias.py``.  

## Environment
### Basic
The two most important packages for experiments are ``transformer_lens`` and ``llama_recipes``. For experiments with GPT2, the enviroment is easy to prepare. While for experiments with Llama, some modifications are made:
* Changes in ``llama_recipes``: Mainly in the training process, for training convenience.
* Changes in ``transformers``: We split the MLP in Llama into MLP heads when performing circuit-tuning.  

We do contain a ```requirement.txt``` file in our codebase, but we recommend you to install the above two packages from source, so it would be easy for development on yourself.  In practice, you can install them by:

```
cd env
bash env.sh
```

P.S.  
Official links:
* llama-recipes: https://github.com/meta-llama/llama-recipes
* TransformerLens: https://github.com/TransformerLensOrg/TransformerLens

### Others
For coreference resolution in ```./data/data_utils_bias.py```, we use the ```neuralcoref``` package, which requires a environment with Python <= 3.7.


## File Structure
* ```./data```: Dataset used in the experiments
    * ```data_utils_xxx.py```: script for data processing
    * ```xxx_dataset.py```: the script need for loading a dataset in the format of [llama-recipes](https://github.com/meta-llama/llama-recipes)
* ```./env```: Environment dependencies
* ```./eap```: A package for Edge Attribution Patching, originated from [EAP](https://github.com/Aaquib111/edge-attribution-patching).
    * ```eap_graph_xxx.py```: Define the computational graph of a model. ```eap_graph.py``` is for GPT2, ```eap_graph_llama.py``` is for Llama, and ```eap_graph_old.py``` is the original definition of the computational graph for GPT2 from [EAP](https://github.com/Aaquib111/edge-attribution-patching). The difference between ```eap_graph_old.py``` and ```eap_graph.py``` is that the former regards ```Q/K/V inputs (x) ``` as upstream nodes, while the latter regards ```Q/K/V vectors (W*x) ``` as upstream nodes.
    * ```eap_graph_wrapper.py```: Define the pipeline of edge attribution patching.
    * ```patching_metrics.py```: Define the patching metrics.
* ```circuit_data.py```: dataset loader and collate func for subject-verb disagreement and the gender de-biasing in GPT2.
* ```circuit_tuning_xxx.py```: Our main method. We provide two versions: one for GPT2 and another for Llama. Note that ```circuit_tuning_old.py``` corresponds to ```eap_graph_old.py``` in ```./eap```.
* ```scripts```: Contains bash scripts for running experiments.
* ```./eval```: Evaluations for the tasks above. 
* ```./sv_analyses```: Methods for analyzing a circuit from various aspects.  
* ```utils.py```: Contains some useful functions. For example, we provide methods to convert a `TransformerLens` format model to a huggingface format model.


## Running the experiments
**Note**: For convenience, the paths in our code may be absolute paths. Please modify the paths to the datasets as well as the models according to your own needs.

For circuit-tuning on GPT2, you can run:
```
cd scripts
bash circuit-tuning.sh
```
You can change the values of arguments inside that script.  
For circuit-tuning on Llamas, you can run:
```
cd scripts
bash circuit-tuning_llama.sh
```