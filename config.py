ngram_config = {
    "n": 2,
    "lowercase": True,           # Convert text to lowercase during preprocessing
    "remove_punctuation": True,  # Remove punctuation during preprocessing
}

no_smoothing = {
    "method_name" : "NO_SMOOTH",
}

add_k = {
    "method_name" : "ADD_K",
    'k': 0.0001
}

stupid_backoff = {
    "method_name" : "STUPID_BACKOFF",
    'alpha': 0.4
}

good_turing = {
    "method_name" : "GOOD_TURING",
}

interpolation = {
    "method_name" : "INTERPOLATION",
    'lambdas': [0.33,0.33,0.34]
}

kneser_ney = {
    "method_name" : "KNESER_NEY",
    'discount': 0.75
}

error_correction = {
    "internal_ngram_best_config" : {
        "method_name" : "ADD_K",
    },

}
