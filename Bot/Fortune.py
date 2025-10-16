import logging
import typing

USE_MARKOVIFY = False

logger = logging.getLogger(__name__)

class BaseFortuneGen:
    def __init__(self, corpus: typing.List[str], state_size: int=2):
        pass

    def generate(self, *args, min_words: int=5, max_words: int=20, tries: int=20, seps: typing.List[str]=[".", "?", "!"], **kwargs) -> str:
        return ''

if USE_MARKOVIFY:
    import markovify

    class MarkovFortuneGen(BaseFortuneGen):
        def __init__(self, corpus: typing.List[str], state_size: int):
            self.model = markovify.combine([markovify.Text(c, state_size=state_size) for c in corpus])

        def generate(self, *args, min_words: int=5, max_words: int=20, tries: int=20, seps: typing.List[str]=[".", "?", "!"], **kwargs) -> str:
            return self.model.make_sentence(min_words=min_words, max_words=max_words, tries=tries, *args, **kwargs)

    FortuneGenerator = MarkovFortuneGen

else:

    import nltk
    from nltk import word_tokenize
    from nltk.util import ngrams
    from collections import defaultdict, Counter
    import random

    nltk.download('punkt')

    class NltkFortuneGen(BaseFortuneGen):
        def __init__(self, corpus: typing.List[str], state_size: int=3):
            corpus = ". ".join(corpus).lower()

            self.starts = set()
            for sentence in corpus.split('.'):
                if not sentence: continue
                split = sentence.lstrip().split()
                if len(split) < 2: continue
                self.starts.add(" ".join(split[:2]))

            self.starts = list(self.starts)
            random.shuffle(self.starts)
            logger.debug(f"starts = {self.starts}")

            self._tokens = word_tokenize(corpus)
            _ngrams = list(ngrams(self._tokens, state_size))
            logger.debug(f"ngrams = {_ngrams}")
            self.model = defaultdict(Counter)

            for _ngram in _ngrams:
                self.model[(_ngram[0], _ngram[1])][_ngram[2]] += 1

        def generate(self, *args, min_words: int=5, max_words: int=25, tries: int=30, seps: typing.List[str]=[".", "?", "!"], **kwargs) -> str:
            for attempt in range(tries):
                try:
                    sentence = self.starts[0].split()
                    self.starts = self.starts[1:] + [self.starts[0]]

                    words = random.choice(list(range(min_words, max_words)))
                    logger.debug(f"Generating {words} words starting with {sentence}")

                    for _ in range(words):
                        most_common = self.model[tuple(sentence[-2:])].most_common()
                        logger.debug(f"most_common={most_common}")
                        next_word = random.choices([x[0] for x in most_common], weights=[x[1] for x in most_common])[0]
                        sentence.append(next_word)

                    logger.debug(f"sentence={sentence}")
                    sentence.reverse()
                    for sep in seps:
                        idx = sentence.index(sep)
                        logger.debug(f"sep={sep}, idx={len(sentence) - idx}")
                        if len(sentence) - idx >= min_words: 

                            if idx > 0:
                                joined = ' '.join(sentence[:idx-1:-1])
                            else:
                                joined = ' '.join(sentence[::-1])

                            separators_order = [item for item in sentence if item in seps]

                            for ssep in seps:
                                joined = joined.replace(ssep, ".")
                            return "{}".join([p.lstrip().capitalize() for p in joined.split(" .")]).format(*[f"{x} " for x in separators_order]).replace(" n't", "n't")

                    logger.debug(f"Sentence too short with all separators: {sentence}")

                except Exception:
                    logger.debug(f"Error generating sentence ({attempt})")
            return "None :c"

    FortuneGenerator = NltkFortuneGen
