from collections import Counter
from typing import List, Tuple
from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd

class NoSmoothing(NGramBase):

    def __init__(self):

        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)
    
    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
            # print("Using Default Smoothing:✅✅✅✅✅✅✅")
            log_prob_sum = 0.0
            for ngram in ngrams:
                i=ngram[:-1]
                count_of_context = self.context_counts.get(i, 0)
                count_of_ngram = self.ngram_counts.get(ngram, 0)
                
                if count_of_ngram == 0:
                    return float('inf') 
                
                if count_of_context == 0:
                    return float('inf')
                
                prob = count_of_ngram / count_of_context
                log_prob_sum = log_prob_sum + np.log(prob)
            return log_prob_sum

class AddK(NGramBase):

    def __init__(self):
        super(AddK, self).__init__()
        self.update_config(add_k)

    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
        # print("Using Add-k Smoothing:✅✅✅✅✅✅✅")
        log_prob_sum = 0.0
        k = add_k['k']
        v = self.vocab_size
        # Ensure that self.vocab_size is defined; it should be the size of your vocabulary.
        for ngram in ngrams:
            # Get the count for the n-gram and its context
            i=ngram[:-1]
            count_of_context = self.context_counts.get(i, 0)
            count_of_ngram = self.ngram_counts.get(ngram, 0)
            
            if (count_of_ngram+k) == 0 or (count_of_context+k*v) == 0:
                    return float('inf')  # Log(0) scenario
            prob = (count_of_ngram + k) / (count_of_context + k * v)
            log_prob_sum = log_prob_sum + np.log(prob)
        return log_prob_sum
    
class StupidBackoff(NGramBase):

    def __init__(self):
        super(StupidBackoff, self).__init__()
        self.update_config(stupid_backoff)
        # Caches for speeding up recursive and repeated lookups.
        self._stupid_backoff_cache = {}
        self._context_count_cache = {}
        self._unigram_count_cache = {}

    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
        """
        Compute the log probability of a sequence of n-grams using stupid backoff smoothing.
        """
        # alpha = self.current_config.get("alpha", 0.4)
        alpha = stupid_backoff['alpha']
        # Clear caches in case compute_probability is called multiple times
        self._stupid_backoff_cache.clear()
        self._context_count_cache.clear()
        self._unigram_count_cache.clear()
        
        log_prob_sum = 0.0
        for ngram in ngrams:
            prob = self._stupid_backoff_probability(ngram, alpha)
            log_prob_sum = log_prob_sum + np.log(prob)
        return log_prob_sum

    def _stupid_backoff_probability(self, ngram: Tuple[str, ...], alpha: float) -> float:
        """
        Recursively compute the probability of an n-gram using stupid backoff smoothing.
        Uses caching to avoid redundant computations.
        """
        key = (ngram, alpha)
        if key in self._stupid_backoff_cache:
            return self._stupid_backoff_cache[key]

        # Base case: unigram probability.
        if len(ngram) == 1:
            word = ngram[0]
            count = self._unigram_count(word)
            total = self.unigram_token_count
            prob = count / total if count > 0 else 1.0 / self.vocab_size
            self._stupid_backoff_cache[key] = prob
            return prob

        # For longer n-grams:
        if len(ngram) == self.n:
            count_ngram = self.ngram_counts.get(ngram, 0)
        else:
            count_ngram = self._get_context_count(ngram)

        # Get the context by dropping the last token.
        context = ngram[:-1]
        if not context:
            count_context = self.unigram_token_count
        elif len(context) == self.n:
            count_context = self.ngram_counts.get(context, 0)
        else:
            count_context = self._get_context_count(context)

        if count_ngram > 0 and count_context > 0:
            prob = count_ngram / count_context
        else:
            # Back off: drop the leftmost token and scale by alpha.
            prob = alpha * self._stupid_backoff_probability(ngram[1:], alpha)

        self._stupid_backoff_cache[key] = prob
        return prob

    def _get_context_count(self, context: Tuple[str, ...]) -> int:
        """
        Given a context (a tuple of tokens), compute its count by summing the counts of all
        full n-grams that begin with this context. Results are cached.
        """
        if context in self._context_count_cache:
            return self._context_count_cache[context]

        count = 0
        context_len = len(context)
        for gram, gram_count in self.ngram_counts.items():
            if gram[:context_len] == context:
                count += gram_count
        self._context_count_cache[context] = count
        return count

    def _unigram_count(self, word: str) -> int:
        """
        Compute the count for a unigram (word). Since we only stored full-length n-grams,
        we approximate the unigram count by summing over all n-grams whose last token is the word.
        Results are cached.
        """
        if word in self._unigram_count_cache:
            return self._unigram_count_cache[word]

        count = 0
        for gram, gram_count in self.ngram_counts.items():
            if gram[-1] == word:
                count = count + gram_count
        self._unigram_count_cache[word] = count
        return count


class GoodTuring(NGramBase):

    def __init__(self):
        super(GoodTuring, self).__init__()
        self.update_config(good_turing)
    
    def compute_probability(self, ngrams: tuple) -> float:
        """
        Computes the cumulative log probability of a sequence of n-grams using Good-Turing smoothing.

        :param ngrams: Tuple of n-grams to evaluate.
        :return: The total log probability.
        """
        total_log_probability = 0.0

        # Precompute overall occurrences and the frequency distribution for efficiency.
        overall_occurrences = sum(self.ngram_counts.values())
        count_frequency = Counter(self.ngram_counts.values())
        max_occurrence = max(self.ngram_counts.values()) if overall_occurrences > 0 else 0
        min_occurrence = min(self.ngram_counts.values()) if overall_occurrences > 0 else 0

        # Compute the log probability for each n-gram.
        for ngram in ngrams:
            current_count = self.ngram_counts.get(ngram, 0)
            prob = self._calculate_ngram_probability(
                current_count,
                overall_occurrences,
                count_frequency,
                max_occurrence,
                min_occurrence
            )
            total_log_probability += np.log(prob)

        return total_log_probability

    def _calculate_ngram_probability(self, count: int, overall_occurrences: int, count_frequency: Counter,
                                    max_occurrence: int, min_occurrence: int) -> float:
        """
        Determines the probability for a single n-gram using Good-Turing smoothing.

        :param count: The frequency count for the n-gram.
        :param overall_occurrences: The total count of all n-grams.
        :param count_frequency: A mapping from frequency values to the number of n-grams with that frequency.
        :param max_occurrence: The highest frequency observed among all n-grams.
        :param min_occurrence: The lowest frequency observed among all n-grams.
        :return: The probability for the given n-gram.
        """
        # Adjust the count using Good-Turing smoothing if it's less than the maximum observed frequency.
        if count < max_occurrence:
            next_count = count + 1
            current_count_freq = count_frequency[count]
            next_count_freq = count_frequency[next_count]
            min_plus_one_freq = count_frequency.get(min_occurrence + 1, 0)

            if current_count_freq == 0:
                adjusted_count = ((min_occurrence + 1) * min_plus_one_freq) / next_count_freq
            else:
                adjusted_count = count
        else:
            adjusted_count = count

        # Determine the probability mass reserved for unseen n-grams.
        unseen_probability_mass = count_frequency[min_occurrence] / overall_occurrences

        # For unseen n-grams (count == 0), use the unseen mass; otherwise, use the adjusted count.
        probability = unseen_probability_mass if count == 0 else adjusted_count / overall_occurrences

        return probability




class Interpolation(NGramBase):

    def __init__(self):
        super(Interpolation, self).__init__()
        self.update_config(interpolation)

    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
        """
        Compute the log probability of a sequence of n-grams using linear interpolation smoothing.
        """
        # print("Goodturing")
        log_prob_sum = 0.0
        for ngram in ngrams:
            p = self._interpolated_probability(ngram)
            log_prob_sum += np.log(p)
        return log_prob_sum

    def _interpolated_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Compute the interpolated probability for an n-gram by combining MLE estimates
        from lower-order models using interpolation weights.
        
        For an n-gram of length L, this method calculates probabilities for each sub n-gram 
        (from unigram up to L-gram), weights them appropriately, and returns the sum.
        
        Parameters:
            ngram: Tuple[str, ...]
                The n-gram for which to compute the probability.
        
        Returns:
            float: The interpolated probability (a small constant is returned if the sum is zero).
        """
        order_length = len(ngram)
        weights = self._fetch_interpolation_weights(order_length)
        prob = self._weighted_probability_sum(ngram, weights)
        
        # Ensure a non-zero probability to avoid issues with log(0) later.
        return prob if prob > 0 else 1e-12


    def _fetch_interpolation_weights(self, order_length: int) -> List[float]:
        """
        Retrieve and normalize the interpolation weights for the current n-gram order.
        
        If the configuration does not provide valid lambda weights (or they do not match
        the expected model order), uniform weights are used. For shorter n-grams, the first
        'order_length' weights are extracted and renormalized.
        
        Parameters:
            order_length: int
                The number of tokens in the current n-gram.
        
        Returns:
            List[float]: A list of normalized weights for the available n-gram orders.
        """
        lambda_weights = interpolation['lambdas']
        
        if lambda_weights is None or len(lambda_weights) != self.n:
            # Default to uniform weights for each sub n-gram order.
            return [1.0 / order_length] * order_length
        
        # Use the first 'order_length' weights and renormalize them.
        selected_weights = lambda_weights[:order_length]
        total_weight = sum(selected_weights)
        
        if total_weight > 0:
            return [w / total_weight for w in selected_weights]
        else:
            return [1.0 / order_length] * order_length


    def _weighted_probability_sum(self, ngram: Tuple[str, ...], weights: List[float]) -> float:
        """
        Compute the weighted sum of MLE probabilities for each sub n-gram.
        
        For i = 1, the unigram probability (last token) is calculated; for i = 2, 
        the bigram probability (last two tokens) is calculated, and so on.
        
        Parameters:
            ngram: Tuple[str, ...]
                The full n-gram from which sub n-grams are derived.
            weights: List[float]
                The list of weights corresponding to each sub n-gram order.
        
        Returns:
            float: The total interpolated probability.
        """
        total_prob = 0.0
        for i in range(1, len(ngram) + 1):
            sub_ngram = ngram[-i:]  # Take the last i tokens.
            mle_prob = self._mle_probability(sub_ngram)
            total_prob += weights[i - 1] * mle_prob
        return total_prob


    def _mle_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Compute the maximum likelihood estimate (MLE) probability for the given n-gram.
        For a unigram, this is count(word) / total_unigram_count.
        For a higher order n-gram, it is count(ngram) / count(context).
        """
        if len(ngram) == 1:
            word = ngram[0]
            count = self._unigram_count(word)
            total = self.unigram_token_count
            if total > 0:
                return count / total
            else:
                return 1.0 / self.vocab_size  # fallback for unseen words
        else:
            count_ngram = self.ngram_counts.get(ngram, 0)
            context = ngram[:-1]
            count_context = self.context_counts.get(context, 0)
            if count_context > 0:
                return count_ngram / count_context
            else:
                return 0.0

    def _unigram_count(self, word: str) -> int:
        """
        Determine the frequency of a single token by summing the counts of all n-grams ending with that token.
        
        Args:
            token (str): The token to retrieve the frequency for.
            
        Returns:
            int: The cumulative frequency of the token.
        """
        count = 0
        for gram, gram_count in self.ngram_counts.items():
            if gram[-1] == word:
                count += gram_count
        return count
    
class KneserNey(NGramBase):

    def __init__(self):
        super(KneserNey, self).__init__()
        self.update_config(kneser_ney)
        # self.discount = self.current_config.get("discount", 0.75)
        self.discount = kneser_ney['discount']
        print(self.discount)
    def compute_probability(self, ngrams: List[Tuple[str, ...]]) -> float:
        """
        Compute the log probability of a sequence of n-grams using Kneser-Ney smoothing.
        """
        log_prob_sum = 0.0
        for ngram in ngrams:
            p = self._kneser_ney_probability(ngram)
            log_prob_sum += np.log(p)
        return log_prob_sum
    
    def _kneser_ney_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Recursively compute the Kneser-Ney probability for the given n-gram.
        """
        # Base case: unigram probability using continuation counts.
        if len(ngram) == 1:
            word = ngram[0]
            cont_count = self._continuation_count(word)
            total_continuations = self._total_continuation_count()
            # Avoid division by zero.
            if total_continuations > 0:
                return cont_count / total_continuations
            else:
                return 1.0 / self.vocab_size

        # Split into context and current word.
        context = ngram[:-1]
        word = ngram[-1]

        # Count for the full n-gram and for the context.
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_context = self.context_counts.get(context, 0)

        # If the context count is zero, back off.
        if count_context == 0:
            return self._kneser_ney_probability(ngram[1:])

        # Calculate the adjusted probability for the current n-gram
        adjusted_count = max(count_ngram - self.discount, 0) / count_context

        # Determine how many unique continuations precede the current n-gram
        num_unique_continuations = self._num_continuations(ngram[:-1])

        # Compute the weight for the backoff component
        backoff_weight = (self.discount * num_unique_continuations) / count_context

        # Retrieve the backoff probability for the next lower n-gram level
        lower_order_prob = self._kneser_ney_probability(ngram[1:])

        # Combine the probabilities from the current n-gram and the backoff
        combined_prob = adjusted_count + backoff_weight * lower_order_prob

        # Ensure the probability is not zero or negative
        if combined_prob <= 0:
            combined_prob = 1e-12

        return combined_prob


    from typing import Tuple, Set, Callable, Iterator

    def _extract_context(self, ngram: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Return the context of an n-gram (all tokens except the final one).
        """
        return ngram[:-1]

    def _extract_last_token(self, ngram: Tuple[str, ...]) -> str:
        """
        Return the last token (word) of the n-gram.
        """
        return ngram[-1]

    def _filter_ngrams(self, condition: Callable[[Tuple[str, ...]], bool]) -> Iterator[Tuple[str, ...]]:
        """
        Yield n-grams that satisfy the given condition.
        """
        for ngram in self.ngram_counts:
            if condition(ngram):
                yield ngram

    def _collect_ngram_pairs(self) -> Set[Tuple[Tuple[str, ...], str]]:
        """
        Gather all unique (context, word) pairs from n-grams with at least two tokens.
        """
        pairs = set()
        for ngram in self.ngram_counts:
            if len(ngram) >= 2:
                context = self._extract_context(ngram)
                word = self._extract_last_token(ngram)
                pairs.add((context, word))
        return pairs

    def _continuation_count(self, word: str) -> int:
        """
        Compute the number of distinct contexts in which the specified word appears.
        (Used for unigram continuation probabilities.)
        """
        unique_contexts = set()
        # Select n-grams where the target word is the final token and there is a valid context.
        for ngram in self._filter_ngrams(lambda ng: len(ng) > 1 and self._extract_last_token(ng) == word):
            unique_contexts.add(self._extract_context(ngram))
        return len(unique_contexts)

    def _total_continuation_count(self) -> int:
        """
        Calculate the total number of unique (context, word) pairs available
        in the n-gram counts (i.e. the total number of bigram types).
        """
        return len(self._collect_ngram_pairs())

    def _get_continuations(self, context: Tuple[str, ...]) -> Set[str]:
        """
        Retrieve the set of unique words that immediately follow the given context.
        """
        continuations = set()
        for ngram in self.ngram_counts:
            if self._extract_context(ngram) == context:
                continuations.add(self._extract_last_token(ngram))
        return continuations

    def _num_continuations(self, context: Tuple[str, ...]) -> int:
        """
        Count the number of unique words that follow the provided context.
        (This count is used to compute the back-off weight for the context.)
        """
        return len(self._get_continuations(context))


if __name__=="__main__":
    ns = KneserNey()
    ns.method_name()

    # ns = KneserNey()
    # ns.method_name()

    corpus = [
        "This is a test sentence.",
        "This sentence is a test.",
        "Another test sentence here."
    ]
    
    # Prepare the corpus (preprocess and tokenize each sentence).
    processed_corpus = ns.prepare_data_for_fitting(corpus)
    
    # Fit the model on the processed corpus.
    ns.fit(processed_corpus)
    
    # # print("\nN-gram counts:")
    # for ngram, count in ns.ngram_counts.items():
    #     # print(f"{ngram}: {count}")
    
    if ns.n > 1:
        # print("\nContext counts:")
        for context, count in ns.context_counts.items():
            print(f"{context}: {count}")
    
    # print("\nVocabulary:", ns.vocab)
    
    # Compute perplexity for a sample sentence.
    sample_text = "This sentence is a test."
    perp = ns.perplexity(sample_text)
    print(f"\nPerplexity for '{sample_text}': {perp}")