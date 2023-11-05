# A Better Hate Speech Detection Model

## Description
Our project aims to advance the detection of hate speech by leveraging a contextual BERT-based language model. We seek to improve upon the lexicon-based classifier used by the European Observatory of Online Hate (EOOH). Our evaluation on two publicly available datasets, Kennedy and Dynabench, demonstrates that our model significantly outperforms the EOOH's classifier, confirming the superiority of contextual language models for this task.

The implementation is done in Python, utilizing NVIDIA A100 GPUs, and incorporates libraries such as PyTorch, PyTorch Lightning, and Huggingface.

## Code Structure

```
.
├── data_utils.ipynb
├── hs_detection.ipynb
├── requirements.txt
└── src
    ├── api_eooh.py
    ├── data_utils.py
    ├── models.py
    ├── train.py
    └── utils.py
```

- `hs_detection.ipynb`: Notebook containing code for training and evaluating the models.
- `data_utils.ipynb`: Notebook for downloading and preprocessing datasets.
- `requirements.txt`: Lists all the required packages.
- `src/api_eooh.py`: Functions to interact with the EOOH API.
- `src/data_utils.py`: Functions for dataset class and dataloader creation.
- `src/models.py`: Defines the `HateSpeechClassifier` model class.
- `src/train.py`: Training function for the model.

## Usage
To use this repository:

1. Install the environment:
    ```bash
    conda create -n aiahs python==3.11.4
    conda activate aiahs
    pip install -r requirements.txt
    ```

2. Run the cells in `data_utils.ipynb` to download and clean the datasets.
3. Execute the cells in `hs_detection.ipynb` to train and evaluate the models.

**Note**: A GPU with CUDA support is required, and Python version should be >= 3.9.

## References
- Kennedy dataset: Chris J Kennedy, Geoff Bacon, Alexander Sahn, and Claudia von Vacano. 2020. "Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application." arXiv preprint arXiv:2009.10277.
- Dynabench dataset & model: Bertie Vidgen, Tristan Thrush, Zeerak Waseem, and Douwe Kiela. 2021. "Learning from the worst: Dynamically generated datasets to improve online hate detection."
