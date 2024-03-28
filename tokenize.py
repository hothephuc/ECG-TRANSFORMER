def check_num_of_beats(chunk, idx_normal):
    return len(set(chunk) & set(idx_normal))