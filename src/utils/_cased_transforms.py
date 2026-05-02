import re
from abc import ABC, abstractmethod

import inflect
import nltk
from flair.data import Sentence
from flair.models import SequenceTagger

__all__ = [
    "DropFileExtensions",
    "DropNonAlpha",
    "DropShortWords",
    "DropSpecialCharacters",
    "DropTokens",
    "DropURLs",
    "DropWords",
    "FilterPOS",
    "FrequencyMinWordCount",
    "ReplaceSeparators",
    "ToLowercase",
    "ToSingular",
]


class BaseTextTransform(ABC):
    """Base class for string transforms."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DropFileExtensions(BaseTextTransform):
    """Remove file extensions from the input text."""

    def __call__(self, text: str) -> str:
        """Remove file extensions from the input text.

        Args:
        ----
            text (str): Text to remove file extensions from.

        """
        text = re.sub(r"\.\w+", "", text)

        return text


class DropNonAlpha(BaseTextTransform):
    """Remove non-alpha words from the input text."""

    def __call__(self, text: str) -> str:
        """Remove non-alpha words from the input text.

        Args:
        ----
            text (str): Text to remove non-alpha words from.

        """
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        return text


class DropShortWords(BaseTextTransform):
    """Remove short words from the input text.

    Args:
    ----
        min_length (int): Minimum length of words to keep.

    """

    def __init__(self, min_length: int) -> None:
        super().__init__()
        self.min_length = min_length

    def __call__(self, text: str) -> str:
        """Remove short words from the input text.

        Args:
        ----
            text (str): Text to remove short words from.

        """
        text = " ".join([word for word in text.split() if len(word) >= self.min_length])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_length={self.min_length})"


class DropSpecialCharacters(BaseTextTransform):
    """Remove special characters from the input text.

    Special characters are defined as any character that is not a word character, whitespace,
    hyphen, period, apostrophe, or ampersand.
    """

    def __call__(self, text: str) -> str:
        """Remove special characters from the input text.

        Args:
        ----
            text (str): Text to remove special characters from.

        """
        text = re.sub(r"[^\w\s\-\.\'\&]", "", text)

        return text


class DropTokens(BaseTextTransform):
    """Remove tokens from the input text.

    Tokens are defined as strings enclosed in angle brackets, e.g. <token>.
    """

    def __call__(self, text: str) -> str:
        """Remove tokens from the input text.

        Args:
        ----
            text (str): Text to remove tokens from.

        """
        text = re.sub(r"<[^>]+>", "", text)

        return text


class DropURLs(BaseTextTransform):
    """Remove URLs from the input text."""

    def __call__(self, text: str) -> str:
        """Remove URLs from the input text.

        Args:
        ----
            text (str): Text to remove URLs from.

        """
        text = re.sub(r"http\S+", "", text)

        return text


class DropWords(BaseTextTransform):
    """Remove words from the input text.

    It is case-insensitive and supports singular and plural forms of the words.
    """

    def __init__(self, words: list[str]) -> None:
        super().__init__()
        self.words = words
        self.pattern = r"\b(?:{})\b".format("|".join(words))

    def __call__(self, text: str) -> str:
        """Remove words from the input text.

        Args:
        ----
            text (str): Text to remove words from.

        """
        text = re.sub(self.pattern, "", text, flags=re.IGNORECASE)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pattern={self.pattern})"


class FilterPOS(BaseTextTransform):
    """Filter words by POS tags.

    Args:
    ----
        tags (list): List of POS tags to remove.
        engine (str): POS tagger to use. Must be one of "nltk" or "flair". Defaults to "nltk".

    """

    def __init__(self, tags: list, engine: str = "nltk") -> None:
        super().__init__()
        self.tags = tags
        self.engine = engine

        if engine == "nltk":
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("punkt", quiet=True)
            self.tagger = lambda x: nltk.pos_tag(nltk.word_tokenize(x))
        elif engine == "flair":
            # post-pone loading the flair tagger to avoid a recent error with flair:
            # NotImplementedError: Cannot copy out of meta tensor; no data! Please use
            # torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module
            # from meta to a different device.
            self.tagger = None

    def __call__(self, text: str) -> str:
        """Filter words by POS tags.

        Args:
        ----
            text (str): Text to remove words with specific POS tags from.

        """
        if self.engine == "nltk":
            word_tags = self.tagger(text)
            text = " ".join([word for word, tag in word_tags if tag not in self.tags])
        elif self.engine == "flair":
            sentence = Sentence(text)

            # post-pone loading the flair tagger to avoid a recent error with flair:
            # NotImplementedError: Cannot copy out of meta tensor; no data! Please use
            # torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module
            # from meta to a different device.
            if self.tagger is None:
                self.tagger = SequenceTagger.load("flair/pos-english-fast").predict

            self.tagger(sentence)
            text = " ".join([token.text for token in sentence.tokens if token.tag in self.tags])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tags={self.tags}, engine={self.engine})"


class FrequencyMinWordCount(BaseTextTransform):
    """Keep only words that occur more than a minimum number of times in the input text.

    If the threshold is too strong and no words pass the threshold, the threshold is reduced to
    the most frequent word.

    Args:
    ----
        min_count (int): Minimum number of occurrences of a word to keep.

    """

    def __init__(self, min_count: int) -> None:
        super().__init__()
        self.min_count = min_count

    def __call__(self, text: str) -> str:
        """Keep only words that occur more than a minimum number of times in the input text.

        Args:
        ----
            text (str): Text to remove infrequent words from.

        """
        if self.min_count <= 1:
            return text

        words = text.split()
        word_counts = {word: words.count(word) for word in words}

        # if nothing passes the threshold, reduce the threshold to the most frequent word
        max_word_count = max(word_counts.values() or [0])
        min_count = max_word_count if self.min_count > max_word_count else self.min_count

        text = " ".join([word for word in words if word_counts[word] >= min_count])

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_count={self.min_count})"


class ReplaceSeparators(BaseTextTransform):
    """Replace underscores and dashes with spaces."""

    def __call__(self, text: str) -> str:
        """Replace underscores and dashes with spaces.

        Args:
        ----
            text (str): Text to replace separators in.

        """
        text = re.sub(r"[_\-]", " ", text)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RemoveDuplicates(BaseTextTransform):
    """Remove duplicate words from the input text."""

    def __call__(self, text: str) -> str:
        """Remove duplicate words from the input text.

        Args:
        ----
            text (str): Text to remove duplicate words from.

        """
        text = " ".join(list(set(text.split())))

        return text


class TextCompose:
    """Compose several transforms together.

    It differs from the torchvision.transforms.Compose class in that it applies the transforms to
    a string instead of a PIL Image or Tensor. In addition, it automatically join the list of
    input strings into a single string and splits the output string into a list of words.

    Args:
    ----
        transforms (list): List of transforms to compose.

    """

    def __init__(self, transforms: list[BaseTextTransform]) -> None:
        self.transforms = transforms

    def __call__(self, text: str | list[str]) -> list[str]:
        """Compose several transforms together.

        Args:
        ----
            text (Union[str, list[str]]): Text to transform.

        """
        if isinstance(text, list):
            text = " ".join(text)

        for t in self.transforms:
            text = t(text)
        return text.split()

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToLowercase(BaseTextTransform):
    """Convert text to lowercase."""

    def __call__(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
        ----
            text (str): Text to convert to lowercase.

        """
        text = text.lower()

        return text


class ToSingular(BaseTextTransform):
    """Convert plural words to singular form."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = inflect.engine().singular_noun

    def __call__(self, text: str) -> str:
        """Convert plural words to singular form.

        Args:
        ----
            text (str): Text to convert to singular form.

        """
        words = text.split()
        for i, word in enumerate(words):
            if not word.endswith("s"):
                continue

            if word[-2:] in ["ss", "us", "is"]:
                continue

            if word[-3:] in ["ies", "oes"]:
                continue

            words[i] = self.transform(word) or word

        text = " ".join(words)

        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def default_vocabulary_transforms() -> TextCompose:
    """Preprocess input text with preprocessing transforms."""
    words_to_drop = [
        "image",
        "photo",
        "picture",
        "thumbnail",
        "logo",
        "symbol",
        "clipart",
        "portrait",
        "painting",
        "illustration",
        "icon",
        "profile",
    ]
    pos_tags = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VBG", "VBN"]

    transforms = []
    transforms.append(DropTokens())
    transforms.append(DropURLs())
    transforms.append(DropSpecialCharacters())
    transforms.append(DropFileExtensions())
    transforms.append(ReplaceSeparators())
    transforms.append(DropShortWords(min_length=3))
    transforms.append(DropNonAlpha())
    transforms.append(ToLowercase())
    transforms.append(ToSingular())
    transforms.append(DropWords(words=words_to_drop))
    transforms.append(FrequencyMinWordCount(min_count=2))
    transforms.append(FilterPOS(tags=pos_tags, engine="flair"))
    transforms.append(RemoveDuplicates())

    transforms = TextCompose(transforms)

    return transforms
