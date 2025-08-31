
# N-gram Language Models and Probabilistic Misspelling Correction

This repository contains an implementation of an N-gram language model and a probabilistic misspelling-correction system developed for the ELL884 course assignment.

## ✨ Features

- **N-gram Language Model**  
  Foundational class building n‑gram frequency counts from a text corpus.

- **Six Smoothing Techniques**  
  Implemented smoothing methods to handle zero-probability n‑grams:
  - No Smoothing (Maximum Likelihood Estimation)
  - Add-k Smoothing
  - Stupid Backoff
  - Good-Turing
  - Interpolation
  - Kneser-Ney

- **Probabilistic Spelling Corrector**  
  - Generates candidate corrections using edit distance (insertions, deletions, transpositions, replacements).
  - Ranks candidates by contextual fitness using the n‑gram language model.
  - Selects corrections by minimizing sentence perplexity.

- **Experimental Heuristics**  
  - Jaccard similarity on character n‑grams.
  - Phonetic similarity (Soundex).

- **Configurable Hyperparameters**  
  All important hyperparameters (e.g., n‑gram order, smoothing constants) are centralized in `config.py`.

## 📊 Performance & Evaluation

- Final Score (custom metric from `grader.py`): **84.86%**
- Perplexity was used as an intrinsic evaluation metric.  
  Example test sentence used: `"Ron Slammed his foot on accelerator"`  
  (Perplexity values for various smoothing methods and orders are reported in the project report.)

## ⚙️ Project Structure

```
.
├── data/
│   ├── train1.txt                # Training corpus 1
│   ├── train2.txt                # Training corpus 2
│   └── misspelling_public.txt    # Test set with "truth&&corrupt" format
├── ngram.py                      # Task 1: Base N-gram class implementation
├── smoothing_classes.py          # Task 2: Implementations of smoothing techniques
├── error_correction.py           # Task 3: SpellingCorrector class logic
├── config.py                     # All hyperparameters for the models
├── grader.py                     # Main script to run and evaluate the model
└── scoring.py                    # Helper script for scoring predictions
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- `numpy`
- `pandas`

Install dependencies:

```bash
pip install numpy pandas
```

### Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### Usage

Run the full pipeline:

```bash
python grader.py
```

What `grader.py` does:
- Loads training and test datasets from `data/`.
- Instantiates and fits the `SpellingCorrector` model on training data.
- Generates predictions for the corrupt sentences in the test set.
- Prints the final score to the console.
- Saves predictions to `predictions_2024AIB2291.csv`.

### Configuration

Edit `config.py` to change:
- `N` (n‑gram order)
- Smoothing-specific hyperparameters (e.g., `k` for add-k)
- Candidate generation limits and heuristic weights

## 🧪 Evaluation

- The codebase contains `grader.py` and `scoring.py` for running evaluations and computing the custom metric used in the course assignment.
- Perplexity-based comparisons across smoothing methods are included in the report.

## 🧑‍💻 Author

**Shreyas Shimpi**

## 📄 License

This project does not include a license file by default. Add an appropriate `LICENSE` file if you plan to publish or share this project publicly.

---

If you want any of these sections expanded (detailed function docs, example runs, README as HTML, adding a CONTRIBUTING guide, or CI configuration), tell me which one and I'll produce it.
