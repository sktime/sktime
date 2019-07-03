class BitWord:
    # Used to represent a word for dictionary based classifiers such as BOSS an BOP.
    # Can currently only handle an alphabet size of <= 4 and word length of <= 16.
    # Current literature shows little reason to go beyond this, but the class will need changes/expansions
    # if this is needed.

    def __init__(self,
                 word=0,
                 length=0):
        self.word = word
        self.length = length
        self.bits_per_letter = 2  # this ^2 == max alphabet size
        self.word_space = 32  # max amount of bits to be stored, max word length == this/bits_per_letter

    def push(self, letter):
        # add letter to a word
        self.word = (self.word << self.bits_per_letter) | letter
        self.length += 1

    def shorten(self, amount):
        # shorten a word by set amount of letters
        self.word = self.right_shift(self.word, amount * self.bits_per_letter)
        self.length -= amount

    def word_list(self):
        # list of input integers to obtain current word
        word_list = []
        shift = self.word_space - (self.length * self.bits_per_letter)

        for i in range(self.length-1, -1, -1):
            word_list.append(self.right_shift(self.word << shift, self.word_space - self.bits_per_letter))
            shift += self.bits_per_letter

        return word_list

    @staticmethod
    def right_shift(left, right):
        return (left % 0x100000000) >> right
