from rwe.prepare_data import build, clean


def test_clean():
    assert clean("  hello \n  world\t") == "hello world"
    assert clean(" = Heading = ") == ""
    assert clean("ab") == ""
    assert clean(None) == ""


def test_build_dedupes_caps_and_excludes():
    sources = {
        "a": iter(["Hello there", "hello there", "Second text", "Third text", "Fourth text"]),
        "b": iter(["Second TEXT", "Only in b", "Held out sentence"]),
    }
    rows = build(sources, {"a": 3, "b": 10}, exclude=["held out sentence"], seed=0)
    texts = sorted(r["text"] for r in rows)
    assert texts == ["Hello there", "Only in b", "Second text", "Third text"]
    src = {r["text"]: r["source"] for r in rows}
    assert src["Only in b"] == "b" and src["Second text"] == "a"


def test_build_is_deterministic_and_shuffled():
    make = lambda: {"a": iter([f"text number {i}" for i in range(50)])}
    r1 = [r["text"] for r in build(make(), {}, seed=1)]
    r2 = [r["text"] for r in build(make(), {}, seed=1)]
    r3 = [r["text"] for r in build(make(), {}, seed=2)]
    assert r1 == r2 and r1 != r3
    assert r1 != [f"text number {i}" for i in range(50)]
