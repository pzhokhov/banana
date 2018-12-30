
_all_words = load_dictionary()

def load_dictionary():
    raise NotImplementedError

def make_crossword(letters):
    possible_words = get_possible_words(_all_words, letters)
    letter_usage = compute_letter_usage(possible_words, letters)
    
    # check that all letters can possibly be used
    for letter, count, used_in in letter_usage:
        if len(used_in) < count:
            letters = dump_letter(u)
            return make_crossword(letters)

    # letters with len(used_in) > count are possible intersections
    while True:
        crossword = _make_trial_crossword(letter_usage.deepcopy(), possible_words)
        if valid_crossword(crossword):
            return crossword

def _make_trial_crossword(letter_usage):
    # possible_words = get_currently_possible_words(letter_usage)
    while True:
        possible_words = get_possible_words(letter_usage)
        intersecting_word_idx, intersecting_letter_idx, letter = choose_intersection(crossword, letter_usage)
        new_word, idx = select_word(possible_words, letter) # maybe with probability related to word length
        crossword.append((new_word, idx, intersecting_word_idx, intersecting_letter_idx))
        reduce_letter_count(new_word, letter_usage)
        if not any([count for _, count, _ in letter_usage]):
            return crossword

def valid_crossword(crossword):
    raise NotImplementedError


def parse_crossword(crossword):
    inserted_words = [] 
    for word, idx, intersecting_word_idx, intersecting_letter_idx in crossword:
        if idx is None:
            x, y, horizontal = 0, 0, True # first word starts at 0, 0 (we will adjust that later if necessary and is horizontal)
        else:
            _, isct_x, isct_y, isct_horizontal = inserted_words[idx]
            if isct_horizontal:
                horizontal = False
                x = isct_x + intersecting_letter_idx
                y = isct_y - idx
            else:
                horizontal = True
                x = isct_x - idx
                y = isct_y + intersecting_letter_idx
            
        inserted_words.append((word, x, y, horizontal)) 


def print_crossword(crossword, stream=sys.stdout):
    max_size = 100
    buffer = [[' '] * max_size] * max_size

    horizontal = True
    word_x, word_y = max_size // 2, max_size // 2
    for word, x, y, horizontal in parse_crossword(crossword):
        for idx, letter in enumerate(word):
            if horizontal:
                buffer[x + idx][y] = letter
            else:
                buffer[x][y + idx] = letter 
    
    for inner_buffer in buffer:
        stream.write(''.join(inner_buffer))
    

def choose_intersection(crossword, letter_usage):
    raise NotImplementedError

def select_word(possible_words, letter):
    raise NotImplementedError

def reduce_letter_count(word, letter_usage):
    raise NotImplementedError




if __name__ == '__main__':
    test()
