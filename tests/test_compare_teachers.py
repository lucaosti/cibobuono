"""
Tests for compare_teachers: venue pairing and automatic checks.
"""

__author__ = "Luca Ostinelli"

from scripts.compare_teachers import consensus, evidence_in_transcript, pair_venues, review_rows, timestamp_ok


def v(name="", address="", **kw):
    return {"name": name, "address": address, "on_site": True, **kw}


def test_pairs_by_fuzzy_name_and_by_address_when_unnamed():
    primary = [v("Tiella", "Piazza Regina Margherita 26"), v("", "Via Sannio 41")]
    secondary = [v("Pizza e fritti magno", "Via Sannio 41"), v("tiella")]
    pairs = pair_venues(primary, secondary)
    assert [(a["name"], b["name"]) for a, b in pairs] == [("Tiella", "tiella"), ("", "Pizza e fritti magno")]


def test_unmatched_venues_are_kept_on_both_sides():
    pairs = pair_venues([v("Clove")], [v("Bonci")])
    assert [(a and a["name"], b and b["name"]) for a, b in pairs] == [("Clove", None), (None, "Bonci")]


def test_each_secondary_venue_pairs_once():
    pairs = pair_venues([v("Crosta"), v("Crosta Milano")], [v("Crosta")])
    assert sum(1 for a, b in pairs if a and b) == 1


def test_unnamed_stalls_pair_by_nearest_timestamp():
    primary = [v(timestamp="5:00"), v(timestamp="9:00")]
    secondary = [v(timestamp="9:20"), v(timestamp="5:10"), v(timestamp="20:00")]
    pairs = pair_venues(primary, secondary)
    assert [(a and a["timestamp"], b and b["timestamp"]) for a, b in pairs] == [
        ("5:00", "5:10"), ("9:00", "9:20"), (None, "20:00"),
    ]


def test_name_variants_pair_only_when_close_in_time():
    near = pair_venues([v("Porchetta Diana Aricina", timestamp="3:00")], [v("Diana Riccina", timestamp="3:20")])
    far = pair_venues([v("Porchetta Diana Aricina", timestamp="3:00")], [v("Diana Riccina", timestamp="12:00")])
    assert near[0][1] is not None
    assert far[0][1] is None
    rows = review_rows("vid", {}, {"venues": [near[0][0]]}, {"venues": [near[0][1]]}, "")
    assert rows[0]["agreement"] == "both_variant"


def test_different_names_close_in_time_stay_apart():
    pairs = pair_venues([v("Clove", timestamp="1:00")], [v("Bonci", timestamp="1:30")])
    assert len(pairs) == 2


def test_different_names_at_the_same_moment_are_one_venue():
    # Reviewed on gold200: the 72B named Dumbo after a figure of speech three seconds later.
    pairs = pair_venues([v("Dumbo", timestamp="9:50")], [v("L'Imortage", timestamp="9:47")])
    assert len(pairs) == 1


def test_evidence_check():
    speech = "allora ragazzi siamo qui da pizza e fritti magno in via sannio"
    assert evidence_in_transcript("Pizza e fritti magno", speech)
    assert not evidence_in_transcript("Entriamo da Bonci a Trionfale", speech)
    assert not evidence_in_transcript("", speech)


def test_timestamp_check():
    assert timestamp_ok("4:42", 1200)
    assert timestamp_ok("1:02:03", None)
    assert not timestamp_ok("45:00", 1200)
    assert not timestamp_ok("", 1200)
    assert not timestamp_ok(None, 1200)


def test_off_site_venues_are_ignored():
    rows = review_rows("vid", {"duration": 600}, {"venues": [v("Bonci", on_site=False)]}, {"venues": []}, "")
    assert rows == []


def test_review_rows_label_agreement():
    rows = review_rows(
        "vid", {"title": "t", "duration": 600},
        {"venues": [v("Clove", timestamp="1:00", evidence="siamo da clove")]},
        {"venues": [v("Clove"), v("Bonellino", timestamp="2:00")]},
        "oggi siamo da clove",
    )
    assert [r["agreement"] for r in rows] == ["both", "secondary_only"]
    assert rows[0]["evidence_in_transcript"] and rows[0]["timestamp_ok"]
    assert not rows[1]["evidence_in_transcript"]


def test_acronym_and_shared_evidence_pair_the_same_venue():
    # Reviewed on gold200: FTG is Frenchie To Go, Diana Aricina / Riccina one porchetta stand.
    acronym = pair_venues([v("FTG", timestamp="2:55")], [v("Frenchie To Go", timestamp="2:51")])
    quote = pair_venues(
        [v("Porchetta Diana Aricina", timestamp="3:05", evidence="Panino preso da Diana Aricina")],
        [v("Diana Riccina", timestamp="3:57", evidence="Panino preso da Diana Aricina")],
    )
    assert len(acronym) == 1 and len(quote) == 1


def test_consensus_keeps_named_union_and_drops_unnamed_and_non_food():
    primary = {"video_type": "venue_review", "venues": [
        v("FTG", "9 Rue du Nil", timestamp="2:55"), v(timestamp="5:00"), v("Dumbo", timestamp="9:50")]}
    secondary = {"video_type": "venue_review", "venues": [
        v("Frenchie To Go", timestamp="2:51"), v("Spurchis", timestamp="2:03")]}
    merged = consensus(primary, secondary)
    assert [(m["name"], m.get("alt_name"), m["sources"]) for m in merged] == [
        ("FTG", "Frenchie To Go", ["primary", "secondary"]),
        ("Dumbo", None, ["primary"]),
        ("Spurchis", None, ["secondary"]),
    ]
    other = {"video_type": "other", "venues": [v("bar della zona")]}
    assert consensus(other, {"video_type": "other", "venues": []}) == []


def test_consensus_takes_the_name_from_the_secondary_when_primary_has_none():
    merged = consensus({"venues": [v(timestamp="1:00")]}, {"venues": [v("Trio", "Linienstraße 13", timestamp="1:10")]})
    assert merged[0]["name"] == "Trio" and merged[0]["address"] == "Linienstraße 13"
