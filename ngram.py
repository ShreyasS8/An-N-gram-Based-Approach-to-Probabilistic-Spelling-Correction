from collections import defaultdict
import numpy as np
import pandas as pd
from typing import List, Tuple
import re
import config
# config.py

class NGramBase:
    def __init__(self):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {}

        # change code beyond this point
        #
        self.current_config = {
            "n" : config.ngram_config['n'],
            "lowercase" : config.ngram_config['lowercase'],
            "remove_punctuation" : config.ngram_config['remove_punctuation'],
            "method_name" : config.add_k['method_name']
        }

        self.n = self.current_config['n']
        self.lowercase = self.current_config['lowercase']
        self.remove_punctuation = self.current_config['remove_punctuation']

        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        
        self.vocab_size = 0
        self.number_of_tokens = 0
        # self.total_unigrams = self.count_total_unigrams()
        self.unigram_token_count = 0

    def method_name(self) -> str:

        return f"Method Name: {self.current_config['method_name']}"

    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        for sentence in data:
            # Create tokens with start and end markers.
            tokens_with_markers = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            # Update vocabulary and token count.
            self.vocab.update(tokens_with_markers)
            self.number_of_tokens += len(tokens_with_markers)
            # Get n-grams from the sentence (markers are added in get_ngrams).
            ngrams = self.get_ngrams(sentence)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                context = ngram[:-1]
                self.context_counts[context] += 1
        self.vocab_size = len(self.vocab)
        self.unigram_token_count = self.unigram_token_counts(data)

    def get_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Create n-grams from a list of tokens by adding start and end markers.
        :param tokens: A list of tokens (without markers).
        :return: A list of n-grams (each n-gram is a tuple of strings).
        """
        tokens_with_markers = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        ngrams = []
        for i in range(len(tokens_with_markers) - self.n + 1):
            ngram = tuple(tokens_with_markers[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def unigram_token_counts(self, data: List[List[str]]) -> None:
        total_tokens = 0
        for sentence in data:
            # Add start and end markers.
            tokens_with_markers = sentence + ["</s>"]
            total_tokens += len(tokens_with_markers)
        return total_tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        return text.split()

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text 

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()

    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
            # print("Using Default Smoothing:✅✅✅✅✅✅✅")
            log_prob_sum = 0.0
            for ngram in ngrams:
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(ngram[:-1], 0)
                if count_ngram == 0 or count_context == 0:
                    return float('inf')  # Log(0) scenario
                prob = count_ngram / count_context
                log_prob_sum += np.log(prob)
            return log_prob_sum
    
    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        # Preprocess and tokenize the text.
        tokens = self.tokenize(self.preprocess(text))
        # Get n-grams with markers.
        ngrams = self.get_ngrams(tokens)
        log_prob_sum = 0.0
        total_ngrams = len(ngrams)
        log_prob_sum = self.compute_probability(ngrams)
        
        if log_prob_sum == float('inf'):
            return float('inf')

        avg_log_prob = log_prob_sum / total_ngrams
        perplexity_value = np.exp(-avg_log_prob)
        return perplexity_value

if __name__ == "__main__":
    tester_ = NGramBase()
    test_sentence = "This, is a ;test sentence."
