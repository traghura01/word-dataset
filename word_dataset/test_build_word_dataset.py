import json
import tempfile
import unittest
from pathlib import Path

import build_word_dataset as dataset


class FormatLinesTests(unittest.TestCase):
    def test_uses_the_part_of_speech_as_the_middle_field(self):
        entries = [("run", "verb", [("move fast", ["I run every day."])])]

        lines = dataset.collect_lines(entries)

        self.assertEqual(lines, ["run = move fast || verb || I run every day."])

    def test_keeps_one_line_when_the_same_word_appears_again(self):
        entries = [
            (
                "bicolor",
                "adjective",
                [("having two colors", ["a bicolor flower", "a bicolored stamp"])],
            ),
            ("bicolor", "noun", [("a flag", ["The bicolor flew."])]),
            ("book", "noun", [("a written work", ["I am reading a book."])]),
        ]

        lines = dataset.collect_lines(entries)

        self.assertEqual(
            lines,
            [
                "bicolor = having two colors || adjective || a bicolor flower",
                "book = a written work || noun || I am reading a book.",
            ],
        )

    def test_skips_examples_that_do_not_use_the_word(self):
        entries = [
            ("dog", "noun", [("a person", ["She is a frump", "she's a real dog"])])
        ]

        lines = dataset.collect_lines(entries)

        self.assertEqual(lines, ["dog = a person || noun || she's a real dog"])

    def test_matches_common_inflections(self):
        entries = [
            ("carry", "verb", [("hold", ["She carried it.", "He carries it."])]),
            ("stop", "verb", [("halt", ["The car stopped."])]),
            ("make", "verb", [("create", ["We are making tea."])]),
        ]

        lines = dataset.collect_lines(entries)

        self.assertEqual(
            lines,
            [
                "carry = hold || verb || She carried it.",
                "stop = halt || verb || The car stopped.",
                "make = create || verb || We are making tea.",
            ],
        )

    def test_keeps_the_three_fields_parseable(self):
        entries = [
            ("apple", "noun", [("a line || break", ["An apple = big\nfruit."])])
        ]

        lines = dataset.collect_lines(entries)

        self.assertEqual(
            lines, ["apple = a line | break || noun || An apple = big fruit."]
        )


class WiktionaryEntriesTests(unittest.TestCase):
    def write_jsonl(self, directory, records):
        path = Path(directory) / "pron.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_reads_modern_examples_and_names_the_part_of_speech(self):
        record = {
            "word": "she",
            "pos": "pron",
            "senses": [
                {
                    "glosses": ["The female person previously mentioned."],
                    "examples": [
                        {"text": "I asked Mary, but she said no.", "type": "example"},
                        {"text": "Old ſhe quote", "type": "quotation", "ref": "1590"},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [record])

            entries = list(dataset.wiktionary_entries([path]))

        self.assertEqual(
            entries,
            [
                (
                    "she",
                    "pronoun",
                    [
                        (
                            "The female person previously mentioned.",
                            ["I asked Mary, but she said no."],
                        )
                    ],
                )
            ],
        )

    def test_uses_the_most_specific_gloss(self):
        record = {
            "word": "and",
            "pos": "conj",
            "senses": [
                {
                    "glosses": ["As a coordinating conjunction.", "Used to connect."],
                    "examples": [{"text": "cats and dogs", "type": "example"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [record])

            entries = list(dataset.wiktionary_entries([path]))

        self.assertEqual(
            entries,
            [("and", "conjunction", [("Used to connect.", ["cats and dogs"])])],
        )

    def test_joins_glosses_when_the_last_one_continues_a_sentence(self):
        record = {
            "word": "the",
            "pos": "article",
            "senses": [
                {
                    "glosses": ["Shows the noun is known,", "...because it was mentioned."],
                    "examples": [{"text": "I read the book.", "type": "example"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, [record])

            entries = list(dataset.wiktionary_entries([path]))

        self.assertEqual(
            entries,
            [
                (
                    "the",
                    "article",
                    [
                        (
                            "Shows the noun is known, because it was mentioned.",
                            ["I read the book."],
                        )
                    ],
                )
            ],
        )

    def test_skips_obsolete_senses_form_of_senses_and_phrases(self):
        records = [
            {
                "word": "the",
                "pos": "pron",
                "senses": [
                    {
                        "glosses": ["Early Modern form of thee."],
                        "tags": ["obsolete"],
                        "examples": [{"text": "made the mad", "type": "example"}],
                    },
                    {
                        "glosses": ["Plural of thee."],
                        "form_of": [{"word": "thee"}],
                        "examples": [{"text": "the the", "type": "example"}],
                    },
                ],
            },
            {
                "word": "each other",
                "pos": "pron",
                "senses": [
                    {
                        "glosses": ["One another."],
                        "examples": [{"text": "See each other.", "type": "example"}],
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_jsonl(directory, records)

            entries = list(dataset.wiktionary_entries([path]))

        self.assertEqual(entries, [])


class BuildDatasetTests(unittest.TestCase):
    def test_writes_sorted_utf8_text_file(self):
        entries = [
            ("you", "pronoun", [("the listener", ["You are kind."])]),
            ("and", "conjunction", [("joins", ["cats and dogs"])]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "word_dataset.txt"

            count = dataset.build_dataset(entries, output_path)

            self.assertEqual(count, 2)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "and = joins || conjunction || cats and dogs\n"
                "you = the listener || pronoun || You are kind.\n",
            )


if __name__ == "__main__":
    unittest.main()
