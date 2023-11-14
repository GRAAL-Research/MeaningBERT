# Article Code Base

Here is our refactored code base to reproduce
our [article](https://www.frontiersin.org/articles/10.3389/frai.2023.1223924/full) results.

## To Reproduce our Article Results

The `src` directory is public to make our results more reproducible. One can reproduce our results by
using the codebase. It was coded in Python 3.11.

Note that this codebase is different from the one used for our article. In our article, we create ten different
train/dev/test splits using a different seed each time, and we also create, on run-time, data augmentation generation
using the seed. Thus, we use the publicly available dataset with the seed `42` in this version. Thus, the sweeps
will have a lower impact on results since they will only alter the model initialization, not the dataset splits.
We can grant access to this code base. It was more complex to release this code version, and the execution was
highly long since we generated many data augmentation examples as per our article procedure.

### Installation

You can first install most of the code dependencies with the `src/requirements.txt` file using `pip` (
e.g. `pip install -Ur src/requirements.txt`). However, some dependencies are difficult to install, as is using
a `requirements.txt` file.

#### Install LENS

To install LENS, one needs to install a modified version of the LENS codebase that fixes PyPi's broken build between
sentence-transformers (torch version), sentencepiece and Pandas. To install a working LENS version, use
`pip install git+https://github.com/davebulaval/LENS`.

#### Install QuestEVal

To install QuestEval, one needs to install a modified version of the QuestEval codebase that fixes PyPi broken build
with
SpaCy. To install a working LENS version, use `pip install git+https://github.com/davebulaval/QuestEval`.

## Execution

To execute all our experimentation, run the sweeps files using a terminal (e.g. `./sweep_data_augmentation.sh`) after
setting Wandb to log all the experimentation. Then, after all the sweeps are complete, run
the `figures_generator/generates_results_tables.py` to create all our LaTeX tables.


