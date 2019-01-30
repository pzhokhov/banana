import numpy as np
import collections
import sys
import random
from copy import copy, deepcopy

def load_dictionary():
    with open('scrabble_words.txt') as f:
        data = f.read().split('\n')
        return [word.strip() for word in data][2:]


_all_words = set(load_dictionary())

CrosswordWord = collections.namedtuple('CrosswordWord', 'word x y horizontal')

class BadCrosswordError(BaseException): pass
class UnderusedLettersError(BaseException): pass

class Crossword():
    def __init__(self):
        self.max_size = 40
        self.buffer = np.ones((self.max_size, self.max_size), dtype=np.uint8) * ord(' ')
        self.words = []
    
    def write(self, stream=sys.stdout):
        for line in self.buffer:
            str_line = line.tobytes().decode('ascii').rstrip()
            if str_line != '':
                stream.write(str_line + '\n')

        if stream != sys.stdout:
            return stream

    def copy(self):
        retval = Crossword()
        retval.buffer = self.buffer.copy()
        retval.words = self.words.copy()
        return retval

    def insert(self, word, commit=True):
        assert isinstance(word, CrosswordWord)
        letters_added = []
        if self.valid_insert(word):
            if commit:
                self.words.append(word)

            for i in range(len(word.word)):
                if word.horizontal:
                    buf_x, buf_y = word.x + i, word.y
                else:
                    buf_x, buf_y = word.x, word.y + i

                if self.buffer[buf_y][buf_x] == ord(' '):
                    letters_added.append(word.word[i])   
                if commit: 
                    self.buffer[buf_y, buf_x] = ord(word.word[i])

        return letters_added        

    def valid_insert(self, word):
        if word.horizontal:
            new_words = _get_new_words(self.buffer, word.word, word.x, word.y)
            connected = _test_connected(self.buffer, word.word, word.x, word.y)
        else:
            new_words = _get_new_words(self.buffer.transpose(), word.word, word.y, word.x)
            connected = _test_connected(self.buffer.transpose(), word.word, word.y, word.x)
         
        return (connected or len(self.words) == 0) and all([w in _all_words for w in new_words]) 

    def get_all_intersectable_letters(self):
        possible_letters = []
        for word in self.words:
            for i in range(len(word.word)):
                if word.horizontal:
                    x,y = word.x + i, word.y
                else:   
                    x,y = word.x, word.y + i
                
                if self._intersectable_letter(x,y, not word.horizontal):
                    possible_letters.append((word.word[i], x,y, not word.horizontal))
        return possible_letters

    def sample_letter_from_crossword(self, possible_words):
        possible_letters = []
        for word in self.words:
            for i in range(len(word.word)):
                if word.horizontal:
                    x,y = word.x + i, word.y
                else:   
                    x,y = word.x, word.y + i
                
                if self._intersectable_letter(x,y, not word.horizontal):
                    possible_letters.append((word.word[i], x,y, not word.horizontal))

        return sample_letter(possible_letters, possible_words)

    def _intersectable_letter(self, x, y, horizontal):
        if horizontal:
            left_ok = (x == 0) or self.buffer[y][x-1] == ord(' ')
            right_ok = (x == self.buffer.shape[1]-1) or self.buffer[y][x+1] == ord(' ')
            return left_ok and right_ok
        else:
            top_ok = (y == 0) or self.buffer[y-1][x] == ord(' ')
            bottom_ok = (y == self.buffer.shape[0]-1) or self.buffer[y+1][x] == ord(' ')
            return top_ok and bottom_ok

    def get_all_next_words(self, words, letter_count):
        insertable_words = []
        letters = self.get_all_intersectable_letters()
        if len(self.words) == 0:
            for word in words:
                word_letter_count = collections.defaultdict(int)
                for letter in word:
                    word_letter_count[letter] += 1
                if all([word_letter_count[l] <= letter_count[l] for l in letter_count]):
                    insertable_words.append(CrosswordWord(word, 0, 0, True))

            return sorted(insertable_words)

        for word in words:
            for letter_tuple in letters:
                if letter_tuple[0] not in word:
                    continue
                idx = word.index(letter_tuple[0])
                x,y,horizontal = letter_tuple[1:]
                if horizontal:
                    x = x - idx
                else:
                    y = y - idx

                letters_used = self.insert(CrosswordWord(word, x, y, horizontal), commit=False)
                new_letter_count = letter_count.copy()
                if len(letters_used) == 0:
                    continue 

                for l in letters_used:
                    new_letter_count[l] -= 1
        
                if any([c < 0 for _, c in new_letter_count.items()]):
                    continue
                insertable_words.append(CrosswordWord(word=word, x=x, y=y, horizontal=horizontal))
        return sorted(insertable_words)


class CrosswordTreeNode():
    def __init__(self, all_words, crossword, remaining_letters):
        self.all_words = all_words
        self.crossword = crossword
        self.remaining_letters = remaining_letters
        self.children = None
        

    def expand_children(self):
        if self.children is not None:
            return
        possible_next_words = self.crossword.get_all_next_words(self.all_words, self.remaining_letters)
        self.children = []
        for word in possible_next_words:
            new_crossword = self.crossword.copy()
            new_remaining_letters = self.remaining_letters.copy()
            letters_used = new_crossword.insert(word, commit=True)
            for letter in letters_used:
                new_remaining_letters[letter] -= 1
            
            self.children.append(CrosswordTreeNode(self.all_words, new_crossword, new_remaining_letters))
            random.shuffle(self.children)

    def search_leaves(self, time_to_stop=None):
        import time
        if self.is_leaf():
            self.remaining_letters = {l : c for l,c in self.remaining_letters.items() if c > 0}
            return [(self.crossword, self.remaining_letters)] 
        
        retval = []
        for child_cw in self.children:
            if time_to_stop is not None and time.time() > time_to_stop:
                break

            leaves = child_cw.search_leaves(time_to_stop)
            if any(leaves) and not any(leaves[0][1]):
                return leaves
            retval += leaves

        return sorted(retval, key=lambda t: sum(t[1].values()))
        
        
    def is_leaf(self):
        self.expand_children()
        return len(self.children) == 0 

def sample_letter(possible_letters, possible_words):
    temperature = 10.0
    import numpy as np
    letter2word_count = [len([w for w in possible_words if l[0] in w]) for l in possible_letters]
    letter2word_count = np.array(letter2word_count) / sum(letter2word_count)
    letter_probs = softmax(-temperature * np.array(letter2word_count))
    return possible_letters[np.random.multinomial(1, letter_probs).argmax()]
    # return possible_letters[np.random.randint(len(possible_letters))]

def softmax(x):
    import numpy as np
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / sum(exp_x)
        
   
def _get_new_words(buf, word, x, y):
        new_words = []
        
        # first, check if have accidentally formed a longer word with existing words in the crossword, 
        # and if so, is it valid
        left = x - 1
        while left >= 0 and buf[y][left] != ord(' '): left -= 1

        right = x + len(word)
        while right < buf.shape[1] and buf[y][right] != ord(' '): right += 1

        actual_word =  \
            buf[y, left+1 : x].tobytes().decode('ascii') + \
            word +  \
            buf[y, x + len(word) : right].tobytes().decode('ascii')
        
        new_words.append(actual_word)
        
        # next, check intersecting words
        for idx in range(len(word)):
            top = y - 1
            while top >= 0 and buf[top][x + idx] != ord(' '): top -= 1
            
            bottom = y + 1
            while bottom < buf.shape[0] and buf[bottom][x + idx] != ord(' '): bottom += 1
            
            new_word = buf[top + 1 : bottom, x + idx].tobytes().decode('ascii')
            if len(new_word) > 1:
                new_words.append(new_word)
  
        return new_words

def _test_connected(buf, word, x, y):
    to_be_overwritten = buf[y, x : x + len(word)]

    if to_be_overwritten.tobytes().decode('ascii').strip() == '':
        return False

    for idx, symbol in enumerate(to_be_overwritten):
        if symbol != ord(' ') and symbol != ord(word[idx]):
            return False

    return True


def make_crossword(letters):
    letter_count = count_letters(letters)
    possible_words = get_possible_words(_all_words, letters)
    letter_used_in = {l: set([w for w in possible_words if l in w]) for l in letters}
    
    # letters with len(used_in) > count are possible intersections
    tries = 0
    while True:
        try:
            tries += 1
            return _make_trial_crossword(possible_words, letter_count.copy(), letter_used_in)
        except BadCrosswordError:
            pass           

def _make_trial_crossword(possible_words, letter_count, letters_used_in):
    # possible_words = get_currently_possible_words(letter_usage)
    max_tries = 30
    crosswords = []

    for tries_outer in range(max_tries):
        crossword = Crossword()
        current_letter_count = letter_count.copy()
        for tries_inner in range(max_tries):

            if len(crossword.words) == 0:
                letter = sample_letter(list(letter_count.items()), possible_words)[0]
                word = random.choice([w for w in possible_words if letter in w])
                x, y = 10, 10
                horizontal = True
            else:    
                letter, x, y, horizontal = crossword.sample_letter_from_crossword(possible_words)
                word, idx = select_word(letters_used_in, current_letter_count, letter) # maybe with probability related to word length

                if horizontal:
                    x = x - idx
                else:
                    y = y - idx

            new_letter_count = current_letter_count.copy()
            letters_used = crossword.insert(CrosswordWord(word, x, y, horizontal), commit=False)
            for l in letters_used:
                new_letter_count[l] -= 1
        
            if any([c < 0 for _, c in new_letter_count.items()]):
                continue
    
            current_letter_count = new_letter_count
            crossword.insert(CrosswordWord(word, x, y, horizontal), commit=True)

            if all([c == 0 for _, c in current_letter_count.items()]):
                return crossword, {}
        
        crosswords.append((crossword, current_letter_count))

    crosswords.sort(key=lambda c: sum(c[1].values()))
    for crossword, remaining_letters in crosswords[:5]:
        print(crossword.write())
        print('Remaining letters:')
        print({l:c for l,c in remaining_letters.items() if c > 0})

    return crosswords[0]
           
   
def select_word(letters_used_in, letter_count, letter):
    possible_words = list(letters_used_in[letter])
    random.shuffle(possible_words)

    '''
    for word in possible_words:
        # this condition can be less restrictive
        if all([letter_count[l] > 0 or l == letter for l in word]):
            idx = random.choice([pos for pos, char in enumerate(word) if char == letter])
            return word, idx
    '''

    for word in possible_words:
        # this condition can be less restrictive
        if letter in word:
            idx = random.choice([pos for pos, char in enumerate(word) if char == letter])
            return word, idx

    raise BadCrosswordError

    


def get_possible_words(words, letters):
    letter_count = count_letters(letters)
    possible_words = set([])
    for word in words:
        word_letter_count = count_letters(word)
        possible = True
        for l, c in word_letter_count.items():
            if c > letter_count[l]:
                possible = False
                break
        if possible:
            possible_words.add(word)

    letters_used = set([])
    if len(possible_words) == 0:
        print('No words possible')
        raise UnderusedLettersError
    
    all_letter_count = None
    for word in possible_words:
        all_letter_count = count_letters(word, all_letter_count)
     
    underused_letters = False   
    for l, c in letter_count.items():
        if c > all_letter_count[l]:
            underused_letters = True
            print('Letter ', l, ' is underused, need to dump')

    # if underused_letters:
    #    print('Possible words: ', possible_words)
    #    raise UnderusedLettersError
        
    return possible_words
        

def count_letters(letters, starting_count=None):
    retval = starting_count or collections.defaultdict(lambda: 0)
    for l in letters:
        retval[l] += 1
    return retval


def test_load_dictionary():
    sorted_words = load_dictionary()
    assert sorted_words[0] == 'AA'
    assert sorted_words[-1] == 'ZZZS'


def test_crossword_insert():
    from io import StringIO
    cw = Crossword()
    cw.insert(CrosswordWord('BREAD', 0, 0, True))
    cw.insert(CrosswordWord('ROAM', 1, 0, False))
    
    reference_cw = \
'''
BREAD
 O
 A
 M
'''

    assert cw.write(StringIO()).getvalue() == reference_cw[1:] # remove first newline

def test_crossword_valid_insert():
    cw = Crossword()
    cw.insert(CrosswordWord('BREAD', 0, 0, True))
    cw.insert(CrosswordWord('ROAM', 1, 0, False))
    
    assert     cw.valid_insert(CrosswordWord('MY', 1, 3, True))
    assert not cw.valid_insert(CrosswordWord('OAF', 1, 1, True))
    assert     cw.valid_insert(CrosswordWord('OAF', 0, 2, True))
    assert not cw.valid_insert(CrosswordWord('DZ', 5, 0, False))
    assert not cw.valid_insert(CrosswordWord('BABY', 10, 10, False))


'''
def test_sample_letter():
    cw = Crossword()
    cw.insert(CrosswordWord('BREAD', 0, 0, True))
    letter, x, y, horizontal = cw.sample_letter()
    
    assert y == 0
    assert 0 <= x and x < 5
    assert horizontal == False
     
def test_sample_letter_two_words():
    cw = Crossword()
    cw.insert(CrosswordWord('BREAD', 0, 0, True))
    cw.insert(CrosswordWord('BABY', 0, 0, False))
    letter, x, y, horizontal = cw.sample_letter()
    
    assert (y == 0 and horizontal == False) or (x == 0 and horizontal == True) 
'''

def test_count_letters():
    word = 'QUICKBROWNFOX'
    ref = {'B': 1, 'C': 1, 'F': 1, 'I': 1, 'K': 1, 'N': 1, 'O': 2, 'R': 1, 'Q': 1, 'U': 1, 'W' : 1, 'X': 1}
    letter_count = count_letters(word)
    assert ref == letter_count

    letter_count = count_letters(word, letter_count)
    ref = {'B': 2, 'C': 2, 'F': 2, 'I': 2, 'K': 2, 'N': 2, 'O': 4, 'R': 2, 'Q': 2, 'U': 2, 'W' : 2, 'X': 2}
    assert ref == letter_count

def test_get_next_possible_words():
    cw = Crossword()
    cw.insert(CrosswordWord('BREAD', 0, 0, True))
    words = set(['BABY', 'BOSS'])
    letters = {'A':2, 'B':2, 'S':2, 'O': 1, 'Y':0}
    next_words = cw.get_all_next_words(words, letters)
    ref_next_words = [CrosswordWord('BOSS', 0, 0, False)]
    assert next_words == ref_next_words
    
    words = set(['BABY', 'BOSS', 'ABS'])
    assert Crossword().get_all_next_words(words, letters) == [
        CrosswordWord('ABS', 0, 0, True),
        CrosswordWord('BOSS', 0, 0, True)
    ]
    

def sample_banana_letters(N):
    import random
    import yaml
    with open('letters.yaml') as f:
        letter_dict = yaml.load(f)
    all_letter_list = []
    for letter, count in letter_dict.items():
        all_letter_list += [letter] * count

    random.shuffle(all_letter_list)
    return all_letter_list[:N]


def search_crosswords(letters, timeout=10):
    import time
    letter_count = count_letters(letters)
    all_words = load_dictionary()
    all_words = get_possible_words(all_words, letter_count)
    tree_root = CrosswordTreeNode(all_words, Crossword(), letter_count)
    leaves = tree_root.search_leaves(time.time() + timeout)
    return leaves
    

if __name__ == '__main__':
    N = 20
    letters = sample_banana_letters(N)
    print(letters)
    leaf_cws = search_crosswords(letters)
    for cw,letters in leaf_cws[:5]:
        print(letters)
        cw.write()

    # letters = ['A', 'B', 'S', 'D']
    # make_crossword(letters)[0].write()
