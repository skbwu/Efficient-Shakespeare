# Efficient Shakespeare: Leveraging Word2Vec for Compute-Limited LLM Training
This repository accompanies the paper "Efficient Shakespeare: Leveraging Word2Vec for Compute-Limited LLM Training", submitted as a mini-project to STATS 305B: Models and Algorithms for Discrete Data (Winter 2025).

**A note on data:** The only data required for this repository is `input.txt`, which contains Andrej Karpathy's Tiny Shakespeare dataset, as provided in Part 1 of HW4.

**A note on compute:** Most of the code was run on a system with one NVIDIA A100 GPU with 40GB of RAM. However, most of the models can in fact be trained on much smaller systems, including Google Colab.

**A note on reproducibility:** All figures in our written report (including supplementary figures) can be exactly reproduced using the code provided in this repository. We also provide the following `.pickle` and `.csv` files containing our raw results in the `reproducibility` directory:**
- `baseline_logs.csv` contains the average train times (seconds per iteration), numbers of trainable parameters, and embedding sizes of each of our character-level-tokenized baseline models.
- `baseline_outputs.pickle` stores a dictionary containing each baseline model's output given each test context prompt (see below).
- `baseline_scores.pickle` contains the mean log-probabilities of pre-trained GPT2-XL generating the above outputs given the context (which we interpret as our "oracle" scores for model performance).
- The files `logs.csv`, `outputs.pickle`, and `scores.pickle` contain the corresponding contents for our non-baseline word-level-tokenized models.
Raw Transformer and Word2Vec model weights were also stored for this project, but it would be unfeasible to upload all of them into GitHub. Nonetheless, please see `Description of Python Scripts` for full reproducibility instructions.

**Descriptions of Jupyter Notebooks**
- `1. EDA + Word2Vec`:  We begin by tokenizing the Mini Shakespeare corpus into words and punctuation marks, computing the appearance counts of each token, the distribution of sentence lengths, and proportions of the corpus captured by various vocabulary sizes (i.e., exploratory data analysis / EDA). Next, we train Word2Vec models at various vocabulary sizes and embedding-vector sizes using the `gensim` package to serve as initial token embeddings for our Transformer models down the line. Then, we visualize the pairwise cosine similarities of our learned Word2Vec embeddings to inform downstream modeling choices and to serve as a sanity-check on the retention of semantic meaning (e.g., checking "Romeo" and "Juliet").
- `2. Training Baseline Models`: Using the Transformer architecture that we completed in Part 1 of HW4, we train character-level-tokenized models with embedding sizes of $192$, $384$, $576$, $768$, and $960$ to serve as baselines for our later word-level-tokenized models.
- `3. Creating Test Prompts + Producing Non-Baseline Model Outputs`: We randomly sample $n=50$ test context prompts from the validation split of our Tiny Shakespeare dataset (i.e., the last $10\%$), where we treat text excerpts between consecutive newline characters as potential test prompts. Next, for each of our $M=45$ word-level-tokenized models, we input each test context prompt and generate output text with a `max_new_tokens` parameter of 3x the length of the test context prompt.
- `4. Producing Baseline Model Outputs`: Using the same $n=50$ test context prompts as above, we generate output text from each of our $B=5$ baseline models using analogous settings as above.
- `5. Evaluating Model Outputs Using GPT2-XL Oracle` contains our routines for computing the mean log-probabilities of a pre-trained GPT2-XL model (i.e., our oracle model) generating the above output texts given each test context prompt, for our baseline and non-baseline models.
- `6. Computing Non-Baseline Model Parameter Counts` and `7. Computing Baseline Model Parameter Counts` contains our scripts for computing the numbers of trainable parameters for each of our non-baseline and baseline models, respectively.
- `8. Generating Results Figures` contains our scripts for generating all results figures in our written report.
  
**Description of Python Scripts**
- `baseline_utils.py` and `utils.py` contain helper functions and our Transformer model classes for our baseline and non-baseline models, respectively.
- `trainer_main.py` is our script for training our word-level-tokenized models.

Please see each of the relevant code files, which contain extensive comments, for additional clarification.
