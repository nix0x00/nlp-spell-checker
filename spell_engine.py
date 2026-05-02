import re
from collections import Counter
from nltk import bigrams
from nltk.corpus import words

class SpellCheckerEngine:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.total_words = 0
        self.vocab = set()

        # Load a backup dictionary of standard English words
        self.general_vocab = set(w.lower() for w in words.words())

        # Train the models upon initialization
        self._train_models()

    def _train_models(self):
        """Reads the corpus and builds the probability models."""
        print(f"Training engine on {self.corpus_path}...")
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: {self.corpus_path} not found. Please ensure the file exists.")
            return

        # Tokenize: Extract all alphabetic words, convert to lowercase
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        self.total_words = len(tokens)

        # Build Unigram Model
        self.unigram_counts = Counter(tokens)
        self.vocab = set(self.unigram_counts.keys())

        # Build Bigram Model
        self.bigram_counts = Counter(bigrams(tokens))

        print("Training complete!")
        print(f"Vocabulary Size: {len(self.vocab)} unique words")

    def get_word_probability(self, word):
        """Calculates probability of each word in corpus: P(w) = count(w) / N"""
        return self.unigram_counts[word] / self.total_words if self.total_words > 0 else 0

    def get_bigram_probability(self, word1, word2):
        """Calculates P(w2 | w1) using Laplace (+1) Smoothing to prevent zero probabilities for valid word pairs that 
        didn't happen to appear next to each other in the corpus... """
        bigram_count = self.bigram_counts[(word1, word2)]
        unigram_count = self.unigram_counts[word1]

        return (bigram_count + 1) / (unigram_count + len(self.vocab))

    def check_sentence(self, sentence):
        """
        Scans a full sentence for both Non-Word and Real-Word (Context) errors.
        Returns a list of flagged errors with suggestions...
        """
        # Use finditer to capture exact character positions and original casing
        matches = list(re.finditer(r'\b[a-zA-Z]+\b', sentence))

        # Create a lowercase list of words for the math engine
        words = [m.group().lower() for m in matches]

        flagged_errors = [] 

        for i, word in enumerate(words):
            # Extract the exact original word and its character spans
            original_case_word = matches[i].group()
            start_pos = matches[i].start()
            end_pos = matches[i].end()

            # 1. Non-Word Error & OOV Detection
            if word not in self.vocab:
                if word in self.general_vocab:
                    flagged_errors.append({
                        'index': i,
                        'original': original_case_word,
                        'start': start_pos,
                        'end': end_pos,
                        'type': 'Out-of-Domain (OOV)',
                        'suggestions': [] 
                    })
                    continue 

                suggestions = self.get_candidates(word)
                flagged_errors.append({
                    'index': i,
                    'original': original_case_word,
                    'start': start_pos,
                    'end': end_pos,
                    'type': 'Non-Word Error',
                    'suggestions': suggestions
                })
                continue 
            # 2. Real-Word Error Detection (Context)
            if i > 0: 
                # Skip context checking for very short functional words
                if len(word) <= 3:
                    continue

                prev_word = words[i-1]
                original_prob = self.get_bigram_probability(prev_word, word)

                # Generate 1-edit distance candidates
                edits1 = self._get_edits_distance_1(word)
                valid_edits1 = {w for w in edits1 if w in self.vocab}

                # Generate and pre-filter 2-edit distance candidates
                valid_edits2 = self._get_valid_edits_distance_2(word)

                # Combine these into one set of valid dictionary words
                valid_edits = valid_edits1.union(valid_edits2)

                best_candidate = None
                # Replace original_word with word
                highest_prob = original_prob

                # 1. Get the raw count of the ORIGINAL bigram
                original_raw_count = self.bigram_counts.get((prev_word, word), 0)

                for candidate in valid_edits:
                    candidate_prob = self.get_bigram_probability(prev_word, candidate)
                    candidate_raw_count = self.bigram_counts.get((prev_word, candidate), 0)

                    # 2. Determine if the candidate passes the Threshold
                    passes_gatekeeper = False

                    if original_raw_count == 0:
                        # TIER 1: The original phrase doesn't exist in the corpus. 
                        if candidate_prob > (original_prob * 3) and candidate_raw_count >= 5:
                            passes_gatekeeper = True

                    elif 1 <= original_raw_count <= 10:
                        # TIER 2: The original phrase is rare.
                        if candidate_prob > (original_prob * 10):
                            passes_gatekeeper = True

                    else:
                        # TIER 3: The original phrase is common.
                        if candidate_prob > (original_prob * 50):
                            passes_gatekeeper = True

                    # 3. If it passes the threshold AND is the highest probability seen so far, save it
                    if passes_gatekeeper and candidate_prob > highest_prob:
                        highest_prob = candidate_prob
                        best_candidate = candidate

                if best_candidate:
                    context_candidates = sorted(
                        valid_edits, 
                        key=lambda c: self.get_bigram_probability(prev_word, c), 
                        reverse=True
                    )[:5]

                    formatted_suggestions = []
                    for c in context_candidates:
                        dist = 1 if c in edits1 else 2
                        formatted_suggestions.append((c, dist, self.get_bigram_probability(prev_word, c)))

                    flagged_errors.append({
                        'index': i,
                        'original': original_case_word,
                        'start': start_pos,
                        'end': end_pos,
                        'type': 'Real-Word Error (Context)',
                        'suggestions': formatted_suggestions
                    })

        return flagged_errors

    # Edit Distance Logic (Damerau-Levenshtein)
    def _get_edits_distance_1(self, word):
        """Generates all possible strings that are exactly 1 edit away..."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return frozenset(deletes + transposes + replaces + inserts)

    def _get_valid_edits_distance_2(self, word):
        """Generates 2-edit strings and instantly filters them to save RAM/CPU overhead."""
        return {e2 for e1 in self._get_edits_distance_1(word) for e2 in self._get_edits_distance_1(e1) if e2 in self.vocab}

    # Candidate Generation & Ranking
    def get_candidates(self, word):
        """Returns a sorted list of suggested corrections for a given word, ranked by their unigram probability in the corpus..."""
        word = word.lower()

        # Case 1: The word is already spelled correctly (exists in vocab)
        if word in self.vocab:
            return [(word, 0, self.get_word_probability(word))] 

        # Case 2: Generate valid words that are 1 edit away
        edits1 = self._get_edits_distance_1(word)
        valid_edits1 = {w for w in edits1 if w in self.vocab}
        if valid_edits1:
            sorted_candidates = sorted(valid_edits1, key=self.get_word_probability, reverse=True)
            return [(c, 1, self.get_word_probability(c)) for c in sorted_candidates[:5]] 

        # Case 3: Generate valid words that are 2 edits away (computationally heavier)
        valid_edits2 = self._get_valid_edits_distance_2(word)

        if valid_edits2:
            sorted_candidates = sorted(valid_edits2, key=self.get_word_probability, reverse=True)
            return [(c, 2, self.get_word_probability(c)) for c in sorted_candidates[:5]]

        # Case 4: No dictionary matches found within 2 edits
        return []

# Testing the Engine
if __name__ == "__main__":
    # Initialize the engine
    engine = SpellCheckerEngine('central_bank_corpus.txt')

    print("\n--- Testing Non-Word Error Correction ---")
    test_word = "imflaton" # Missing 'i'
    print(f"Misspelled word: {test_word}")
    suggestions = engine.get_candidates(test_word)

    for word, distance, prob in suggestions:
        print(f"Suggestion: {word:<15} | Edit Distance: {distance} | Probability: {100 * prob:.4f}%")
    print("\n--- Testing Bigram (Context) Probabilities ---")

    # test a very common phrase in central banking
    word1 = "interest"
    word2_good = "rates"
    word2_bad = "banana"

    prob_good = engine.get_bigram_probability(word1, word2_good)
    prob_bad = engine.get_bigram_probability(word1, word2_bad)

    print(f"Context: What is the probability that '{word2_good}' comes right after '{word1}'?")
    print(f"Result: {100 * prob_good:.4f}%")

    print(f"\nContext: What is the probability that '{word2_bad}' comes right after '{word1}'?")
    print(f"Result: {100 * prob_bad:.4f}%")

    print("\n--- Testing Sentence Context Checker ---")

    # "graffe" is a non-word error. 
    # "rats" is a valid word, but a real-word context error following "interest".
    test_sentence = "The central bank raised the interest rats because of the graffe."

    print(f"Input: '{test_sentence}'\n")

    results = engine.check_sentence(test_sentence)

    for error in results:
        print(f"[{error['type']}] Word: '{error['original']}'")
        print(f"Top Suggestion: {error['suggestions'][0][0]} (Edit Distance: {error['suggestions'][0][1]} | Prob: {error['suggestions'][0][2]*100:.4f}%)")
        print("-" * 40)
