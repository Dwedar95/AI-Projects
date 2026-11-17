import pytest

from rag_eval_kit.chunking import (
    FixedWindowChunker,
    HeadingChunker,
    SentenceChunker,
    get_chunker,
    pack_sentences,
    split_sections,
)
from rag_eval_kit.documents import Document, parse_markdown
from rag_eval_kit.text import split_sentences, stem, tokenize

DOC = parse_markdown(
    "doc",
    "# Title\n\nIntro sentence one. Intro sentence two.\n\n"
    "## Alpha\nFirst alpha. Second alpha is longer than the first.\n\n"
    "## Beta\n- item one\n- item two\n",
)


def test_tokenize_removes_stopwords_stems_and_keeps_identifiers():
    assert tokenize("The devices are restarting with error E-214!") == ["devic", "restart", "error", "e-214"]
    assert stem("policies") == "policy" and stem("access") == "access" and stem("mw-200") == "mw-200"


def test_split_sentences_handles_lists_headings_and_abbreviated_numbers():
    text = "## Head\nCosts are 0.30 euros per km. Second sentence!\n- bullet one\n- bullet two"
    assert split_sentences(text) == [
        "Head",
        "Costs are 0.30 euros per km.",
        "Second sentence!",
        "bullet one",
        "bullet two",
    ]


def test_parse_markdown_extracts_title():
    assert DOC.title == "Title"
    assert not DOC.text.startswith("# Title")


def test_fixed_window_overlap_and_coverage():
    document = Document("d", "T", " ".join(f"w{i}" for i in range(25)))
    chunks = FixedWindowChunker(size=10, overlap=4).chunk(document)
    assert [c.text.split()[0] for c in chunks] == ["w0", "w6", "w12", "w18"]
    assert chunks[0].text.split()[-4:] == chunks[1].text.split()[:4]
    assert chunks[-1].text.split()[-1] == "w24"
    assert [c.chunk_id for c in chunks] == ["d::0", "d::1", "d::2", "d::3"]
    with pytest.raises(ValueError):
        FixedWindowChunker(size=5, overlap=5)


def test_pack_sentences_respects_budget_and_overlap():
    sentences = ["a b c", "d e f", "g h i", "j k l"]
    assert pack_sentences(sentences, max_words=6, overlap_sentences=0) == ["a b c d e f", "g h i j k l"]
    assert pack_sentences(sentences, max_words=6, overlap_sentences=1) == ["a b c d e f", "d e f g h i", "g h i j k l"]
    assert pack_sentences(["one very long sentence here"], max_words=2, overlap_sentences=1) == [
        "one very long sentence here"
    ]


def test_sentence_chunker_never_cuts_a_sentence():
    chunks = SentenceChunker(max_words=8, overlap_sentences=0).chunk(DOC)
    sentences = split_sentences(DOC.text)
    assert " ".join(chunk.text for chunk in chunks) == " ".join(sentences)
    boundaries = {" ".join(sentences[:i]) for i in range(1, len(sentences) + 1)}
    joined = ""
    for chunk in chunks:
        joined = f"{joined} {chunk.text}".strip()
        assert joined in boundaries


def test_heading_chunker_keeps_sections_and_names_them():
    assert [heading for heading, _ in split_sections(DOC.text)] == ["", "Alpha", "Beta"]
    chunks = HeadingChunker().chunk(DOC)
    assert [c.section for c in chunks] == ["", "Alpha", "Beta"]
    assert chunks[1].text == "First alpha. Second alpha is longer than the first."
    assert chunks[1].indexed_text.startswith("Title. Alpha.")
    assert len(HeadingChunker(max_words=4, overlap_sentences=0).chunk(DOC)) > 3


def test_get_chunker_rejects_unknown_strategy():
    assert isinstance(get_chunker("fixed"), FixedWindowChunker)
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        get_chunker("semantic")
