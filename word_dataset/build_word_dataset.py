"""Build a dictionary training file for every English word with examples.

Nouns, verbs, adjectives and adverbs come from Princeton WordNet.
Pronouns, determiners, articles, conjunctions, prepositions and particles
come from English Wiktionary (CC BY-SA) via the kaikki.org extracts.

Each saved line uses this format:

    word = definition || part of speech || sentence
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

MAX_SENTENCES = 2

WORDNET_PARTS = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb"}

WIKTIONARY_PARTS = {
    "pron": "pronoun",
    "det": "determiner",
    "article": "article",
    "conj": "conjunction",
    "prep": "preposition",
    "particle": "particle",
}

WIKTIONARY_URL = (
    "https://kaikki.org/dictionary/English/pos-{pos}/"
    "kaikki.org-dictionary-English-by-pos-{pos}.jsonl"
)

SKIPPED_SENSE_TAGS = {
    "obsolete",
    "archaic",
    "dialectal",
    "rare",
    "nonstandard",
    "misspelling",
}

WORD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z']*$")
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
CONTRACTIONS = ("'s", "'ll", "'m", "'d", "'ve", "'re")


def collect_lines(entries):
    return _first_line_per_word(_collect_groups(entries))


def build_dataset(entries, output_path):
    groups = _collect_groups(entries)
    groups.sort(key=lambda group: (group[0].casefold(), group[0], group[1]))
    lines = _first_line_per_word(groups)
    output_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(lines)


def wordnet_entries():
    from nltk.corpus import wordnet

    for code, part in WORDNET_PARTS.items():
        for name in sorted(wordnet.all_lemma_names(pos=code)):
            if not WORD_PATTERN.match(name):
                continue
            senses = [
                (synset.definition(), synset.examples())
                for synset in wordnet.synsets(name, pos=code)
                if name.casefold()
                in (lemma.casefold() for lemma in synset.lemma_names())
            ]
            if senses:
                yield name, part, senses


def wiktionary_entries(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for raw_line in handle:
                record = json.loads(raw_line)
                word = record.get("word", "")
                part = WIKTIONARY_PARTS.get(record.get("pos"))
                if not part or not WORD_PATTERN.match(word):
                    continue
                senses = [
                    sense
                    for sense in map(_wiktionary_sense, record.get("senses", []))
                    if sense
                ]
                if senses:
                    yield word, part, senses


def download_wiktionary(folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for pos in WIKTIONARY_PARTS:
        path = folder / f"wiktionary-{pos}.jsonl"
        if not path.is_file():
            print(f"Downloading Wiktionary {WIKTIONARY_PARTS[pos]} entries...")
            urllib.request.urlretrieve(WIKTIONARY_URL.format(pos=pos), path)
        paths.append(path)
    return paths


def ensure_wordnet():
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Save definitions, parts of speech and example sentences."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file. Defaults to word_dataset.txt.",
    )
    args = parser.parse_args(argv)

    folder = Path(__file__).resolve().parent
    output_path = args.output or folder / "word_dataset.txt"

    ensure_wordnet()
    wiktionary_paths = download_wiktionary(folder / "data")

    def all_entries():
        yield from wordnet_entries()
        yield from wiktionary_entries(wiktionary_paths)

    count = build_dataset(all_entries(), output_path)
    print(f"Saved {count} lines to {output_path}")
    return 0


def _first_line_per_word(groups):
    word_counts = {}
    lines = []
    for word, _part, group_lines in groups:
        key = word.casefold()
        if key in word_counts:
            word_counts[key] += 1
        else:
            word_counts[key] = 1
            if group_lines:
                lines.append(group_lines[0])
    return lines


def _collect_groups(entries):
    senses_by_key = {}
    for word, part, senses in entries:
        senses_by_key.setdefault((word, part), []).extend(senses)

    groups = []
    for (word, part), senses in senses_by_key.items():
        lines = _pick_lines(word, part, senses)
        if lines:
            groups.append((word, part, lines))
    return groups


def _pick_lines(word, part, senses):
    usable = []
    for definition, examples in senses:
        clean_definition = _clean_field(definition)
        sentences = [
            sentence
            for sentence in map(_clean_field, examples)
            if sentence and _example_uses_word(sentence, word)
        ]
        if clean_definition and sentences:
            usable.append((clean_definition, sentences))

    lines = []
    round_index = 0
    while len(lines) < MAX_SENTENCES and any(
        round_index < len(sentences) for _definition, sentences in usable
    ):
        for definition, sentences in usable:
            if round_index < len(sentences) and len(lines) < MAX_SENTENCES:
                lines.append(f"{word} = {definition} || {part} || {sentences[round_index]}")
        round_index += 1
    return lines


def _wiktionary_sense(sense):
    if sense.get("form_of") or sense.get("alt_of"):
        return None
    if SKIPPED_SENSE_TAGS.intersection(sense.get("tags", [])):
        return None
    glosses = sense.get("glosses") or []
    if not glosses:
        return None
    examples = [
        example.get("text", "")
        for example in sense.get("examples", [])
        if example.get("type") != "quotation"
        and not example.get("ref")
        and "ſ" not in example.get("text", "")
    ]
    if not examples:
        return None
    gloss = glosses[-1]
    if gloss.startswith("..."):
        gloss = " ".join(part.lstrip(".") for part in glosses)
    return gloss, examples


def _clean_field(value):
    return " ".join(value.split()).replace("||", "|")


def _example_uses_word(example, word):
    forms = _word_forms(word.casefold())
    for token in TOKEN_PATTERN.findall(example.replace("\u2019", "'")):
        if token.casefold().strip("'") in forms:
            return True
    return False


def _word_forms(word):
    forms = {word, word + "s", word + "es", word + "d", word + "ed", word + "ing"}
    forms.update(word + suffix for suffix in CONTRACTIONS)
    if word.endswith("y"):
        forms.update({word[:-1] + "ies", word[:-1] + "ied"})
    if word.endswith("e"):
        forms.add(word[:-1] + "ing")
    if len(word) > 2:
        forms.update({word + word[-1] + "ed", word + word[-1] + "ing"})
    return forms


if __name__ == "__main__":
    raise SystemExit(main())
