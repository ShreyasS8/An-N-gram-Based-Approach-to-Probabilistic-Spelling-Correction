import re
# from tqdm import tqdm
from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd

# train_files = [r"data\train1.txt", r"data\train2.txt"]
# vocab_file = "words_list.txt"
# output_vocab = "combined_vocab.txt"

class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']
        
        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])
        
        self.vocab1 = self.internal_ngram.vocab
        
                    
    def fit(self, data: List[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)

    def correct(self, text: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        ## there will be an assertion to check if the output text is of the same
        ## length as the input text
        corrected_texts = []
        edit_distance_max: int = 1
        
        for sentence in text:
            # Use the same fixed preprocessing and tokenization as used in training.
            tokens = self.internal_ngram.fixed_tokenize(self.internal_ngram.fixed_preprocess(sentence))
            v=self.internal_ngram.vocab
            corrected_tokens = tokens.copy()
            # Compute baseline perplexity for the original sentence.
            baseline_perp = self.internal_ngram.perplexity(" ".join(tokens))
            # For each token, if it is not in the learned vocabulary, try to correct it.
            for i, token in enumerate(tokens):
                if token not in self.vocab1:
                    candidate_tokens = self.candidates(token, edit_distance_max)
                    if candidate_tokens:
                        best_perp = baseline_perp  # Use baseline as the reference perplexity.
                        best_candidate = token
                        
                        # Try each candidate correction.
                        for candidate in candidate_tokens:
                            temp_tokens = corrected_tokens.copy()
                            temp_tokens[i] = candidate
                            candidate_sentence = " ".join(temp_tokens)
                            candidate_sentence = candidate_sentence
                            perplexity_score = self.internal_ngram.perplexity(candidate_sentence)
                            # Only update if the candidate improves the perplexity.
                            if perplexity_score < best_perp:
                                best_candidate = candidate
                                best_perp = perplexity_score
                                
                        corrected_tokens[i] = best_candidate
            corrected_text = " ".join(corrected_tokens)
            corrected_texts.append(corrected_text)
        return corrected_texts

    def edits1(self, word: str) -> set[str]:
        """
        Generate all strings that are one edit away from the input word.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits_edit = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes_edit = [L + R[1:] for L, R in splits_edit if R]
        transposes_edits = [L + R[1] + R[0] + R[2:] for L, R in splits_edit if len(R) > 1]
        replaces_edits = [L + c + R[1:] for L, R in splits_edit if R for c in letters]
        inserts_edits = [L + c + R for L, R in splits_edit for c in letters]
        return set(deletes_edit + transposes_edits + replaces_edits + inserts_edits)
    
    def all_edits(self, word: str, max_distance: int) -> set[str]:
        """
        Generate all strings that are within a given edit distance from the input word.
        This function uses an iterative approach to generate edits at distances 1 up to max_distance.
        """
        edits = set()
        current_edits = {word}
        for _ in range(max_distance):
            new_edits = set()
            for w in current_edits:
                new_edits.update(self.edits1(w))
            edits.update(new_edits)
            current_edits = new_edits
        return edits
    
    def candidates(self, word: str, max_distance: int = 2) -> set[str]:
        """
        Generate candidate corrections for a misspelled word that exist in the model’s vocabulary.
        :param word: The misspelled word.
        :param max_distance: Maximum edit distance to consider.
        :return: A set of candidate corrections.
        """
        vocab1 = self.vocab1
        if word in vocab1:
            return {word}
        cand = set(w for w in self.all_edits(word, max_distance) if w in vocab1)
        return cand