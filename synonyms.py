import math

"""
Semantic Similarity Synonyms Finder
Author: [Tannaz Chowdhury]
Date: [11/2024]
Course: ESC180 - Introduction to Computer Programming

Description:
This program determines the semantic similarity between words using 
vector-based representations built from text corpora. It implements 
cosine similarity to answer TOEFL-style synonym questions, achieving 
up to ~70% accuracy. 

Key Features:
- Constructs semantic descriptors using word co-occurrence in sentences.
- Computes similarity between words using cosine similarity.
- Selects the most similar word from multiple-choice options.

"""

def norm(vec):
    # Return the norm of a vector stored as a dictionary, as described in the handout for Project 3.
    # The norm is calculated as the square root of the sum of squares of the vector values.
    # Parameters:
    #     vec (dict): Dictionary representing a vector, where keys are components and values are magnitudes.
    # Returns:
    #     float: The norm of the vector.
    sum_of_squares = 0.0
    for x in vec:
        sum_of_squares += vec[x] * vec[x]
    return math.sqrt(sum_of_squares)

def cosine_similarity(vec1, vec2):
    # Calculate cosine similarity between two sparse vectors stored as dictionaries.
    # This function returns the cosine similarity between the sparse vectors vec1 and vec2, stored as dictionaries.
    # Parameters:
    #     vec1 (dict): The first vector represented as a dictionary.
    #     vec2 (dict): The second vector represented as a dictionary.
    # Returns:
    #     float: The cosine similarity between vec1 and vec2. Returns -1.0 if either vector has a norm of zero.
    # Example:
    #     cosine_similarity({"a": 1, "b": 2, "c": 3}, {"b": 4, "c": 5, "d": 6}) should return approximately 0.70.
    
    # Initialize components for the cosine similarity calculation
    base1 = 0
    base2 = 0
    top = 0

    # Calculate base values (norms) and the dot product
    for key, value in vec1.items():
        base1 += value ** 2
    for key, value in vec2.items():
        base2 += value ** 2
    for key1, value1 in vec1.items():
        if key1 in vec2:
            top += value1 * vec2[key1]

    # Calculate the cosine similarity
    if base1 == 0 or base2 == 0:
        return -1.0  # Use -1.0 to indicate no valid similarity can be computed
    return top / (base1 ** 0.5 * base2 ** 0.5)

def build_semantic_descriptors(sentences):
    # Build semantic descriptors from a list of sentences.
    # This function takes in a list of sentences, where each sentence is a list of words (strings),
    # and returns a dictionary where each word is a key and its value is another dictionary that represents
    # the semantic descriptor of that word.
    # Parameters:
    #     sentences (list): A list of sentences, each sentence being a list of words.
    # Returns:
    #     dict: A dictionary of semantic descriptors for each word in the sentences.
    # Example:
    #     If sentences represents the opening of Notes from the Underground above, the dictionary returned would have keys for each word with corresponding co-occurrence counts.
    descriptors = {}

    for sentence in sentences:
        unique_words = set(sentence)
        for word in unique_words:
            if word not in descriptors:
                descriptors[word] = {}
            for other_word in unique_words:
                if other_word != word:
                    descriptors[word][other_word] = descriptors[word].get(other_word, 0) + 1
    
    return descriptors

def build_semantic_descriptors_from_files(filenames):
    # Build semantic descriptors from multiple text files.
    # This function takes a list of filenames, reads the text in each file, processes the text by removing certain punctuation,
    # and splits the text into sentences to create semantic descriptors for all the words in the files.
    # Parameters:
    #     filenames (list): List of strings, where each string is a filename.
    # Returns:
    #     dict: A dictionary of semantic descriptors for each word in the text files.
    # Assumptions:
    #     Only the following punctuation separates sentences: '.', '!', '?'.
    #     All other punctuation is replaced with spaces.
    all_sentences = []
    
    for filename in filenames:
        with open(filename, "r", encoding="latin1") as file:
            text = file.read().lower()
            
            # Replace punctuation and split into sentences
            text = text.replace("--", " ").replace(",", " ").replace("-", " ").replace(":", " ").replace(";", " ")
            text = text.replace("!", ".").replace("?", ".")
            sentences = [s.strip().split() for s in text.split(".") if s.strip()]

            all_sentences.extend(sentences)
    
    return build_semantic_descriptors(all_sentences)

def most_similar_word(word, choices, semantic_descriptors, similarity_fn):
    # Find the most similar word among choices using semantic descriptors.
    # This function takes in a word, a list of choices, and a dictionary of semantic descriptors, and returns
    # the element of choices which has the largest semantic similarity to the word, computed using the given similarity function.
    # Parameters:
    #     word (str): The word to find the most similar word for.
    #     choices (list): List of strings representing the possible choices.
    #     semantic_descriptors (dict): Dictionary containing semantic descriptors for words.
    #     similarity_fn (function): A function that calculates the similarity between two vectors represented as dictionaries.
    # Returns:
    #     str: The choice with the highest similarity score to the given word.
    max_similarity = -1
    best_choice = choices[0]

    for choice in choices:
        if word not in semantic_descriptors or choice not in semantic_descriptors:
            similarity = -1
        else:
            similarity = similarity_fn(semantic_descriptors[word], semantic_descriptors[choice])

        if similarity > max_similarity:
            max_similarity = similarity
            best_choice = choice

    return best_choice

def run_similarity_test(filename, semantic_descriptors, similarity_fn):
    # Run similarity test on a file and return percentage of correct answers.
    # This function takes in a filename containing test questions, a dictionary of semantic descriptors,
    # and a similarity function, and returns the percentage of correct answers.
    # Parameters:
    #     filename (str): The name of a file containing test questions in the specified format.
    #     semantic_descriptors (dict): Dictionary containing semantic descriptors for words.
    #     similarity_fn (function): A function that calculates the similarity between two vectors represented as dictionaries.
    # Returns:
    #     float: The percentage of questions correctly answered (between 0.0 and 100.0).
    # Example:
    #     Each line in the file represents a question in the form: 'word correct_answer choice1 choice2 ...'.
    total_count = 0
    correct_count = 0

    with open(filename, "r", encoding="latin1") as file:
        for line in file:
            words = line.strip().split()
            if len(words) >= 3:
                word, correct_answer = words[0], words[1]
                choices = words[2:]
                guess = most_similar_word(word, choices, semantic_descriptors, similarity_fn)
                if guess == correct_answer:
                    correct_count += 1
                total_count += 1

    if total_count == 0:
        return 0.0
    return (correct_count / total_count) * 100

# Test code
if __name__ == "__main__":
    # Test build_semantic_descriptors
    S = [
        ["i", "am", "a", "sick", "man"],
        ["i", "am", "a", "spiteful", "man"],
        ["i", "am", "an", "unattractive", "man"]
    ]
    
    descriptors = build_semantic_descriptors(S)
    print("Semantic descriptors for 'am':", descriptors.get("am", {}))
    
    # Test cosine_similarity
    vec1 = {"a": 1, "b": 2, "c": 3}
    vec2 = {"b": 4, "c": 5, "d": 6}
    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine similarity: {similarity:.2f}")
    
    # Test semantic descriptor building from files and run similarity test
    sem_descriptors = build_semantic_descriptors_from_files(["wp.txt", "sw.txt"])
    res = run_similarity_test("test.txt", sem_descriptors, cosine_similarity)
    #print(f"{res}% of the guesses were correct")
   
