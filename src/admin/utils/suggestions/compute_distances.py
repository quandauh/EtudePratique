import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_distances(answers, characters, questions, weightMode = False):
    """
    Compute the cosine similarity between characters based on their answers to questions.
    """

    # We bring back indexes for characters and questions from 0 to n (they are not continuous)
    character_index = {character: idx for idx, character in enumerate(characters)}
    question_index = {question: idx for idx, question in enumerate(questions)}

    # Initialize the answers matrix with zeros
    combined_matrix = np.zeros((len(characters), 2 * len(questions)))
    
    for answer in answers:
        char_idx = character_index[answer.item_id]
        ques_idx = question_index[answer.question_id]
        combined_matrix[char_idx][ques_idx] = answer.yes_tfidf
        combined_matrix[char_idx][len(questions) + ques_idx] = answer.no_tfidf

    # Compute weights for each question if weightMode is True
    # Here, we use the average TF-IDF scores across all characters to determine the importance of each question
    if(weightMode):    
        average_yes_tfidf = np.mean(combined_matrix[:, :len(questions)], axis=0)
        average_no_tfidf = np.mean(combined_matrix[:, len(questions):], axis=0)
        # Combine the weights for yes and no answers into a single array
        weights = np.hstack([average_yes_tfidf, average_no_tfidf])
        
        weights /= weights.sum()  # Normalize the weights
        combined_matrix = combined_matrix * weights

    # Compute the cosine similarity
    similarity_matrix = cosine_similarity(combined_matrix)
    #print(similarity_matrix)
    return similarity_matrix, character_index, question_index, combined_matrix