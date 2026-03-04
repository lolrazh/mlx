#!/usr/bin/env python3
"""Apply review fixes to V5 raw data.

Fixes identified by 5 Opus review agents:
  - multistep-spellcaps: 4 fixes (acronyms + nonsense)
  - multistep-quote-corr: 5 fixes (@-symbol placement + pronoun)
  - multistep-corr-format: 3 fixes (scope + consistency)
  - multistep-complex: 6 fixes (Golden Rule + emphasis + self-correction)
  - spell-casual: 2 fixes (trigger phrase + non-error)
  - spell-compound-scope: 5 fixes (CamelCase splitting)
  - emphasis-caps: 3 fixes (sentence merging + dropped article)
  - spell-alt-phrase: DELETE entirely (all 7 are phonetic breakdowns)

Run: python spoke/data/v5/fix_review.py
"""

import json
from pathlib import Path

RAW = Path(__file__).parent / "raw"


def load(name):
    with open(RAW / name) as f:
        return json.load(f)


def save(name, data):
    with open(RAW / name, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    total_fixed = 0
    total_deleted = 0

    # ── 1. multistep-spellcaps: fix acronyms + delete nonsense ──
    data = load("multistep-spellcaps.json")

    # #5 (idx 4): CDG → should stay CDG, not Cdg
    if "Cdg" in data[4]["ideal"]:
        data[4]["ideal"] = "We're flying into CDG."
        total_fixed += 1

    # #6 (idx 5): FIFTYNINETY is nonsensical — replace entire example
    if "FIFTYNINETY" in data[5]["ideal"]:
        data[5] = {
            "input": "The wine region is called Bordo. Spell that B-O-R-D-E-A-U-X, all caps.",
            "ideal": "The wine region is called BORDEAUX."
        }
        total_fixed += 1

    # #7 (idx 6): IFC → should stay IFC, not Ifc
    if "Ifc" in data[6]["ideal"]:
        data[6]["ideal"] = "She works at the World Bank in their IFC division."
        total_fixed += 1

    # #10 (idx 9): DKA → should stay DKA, not Dka
    if "Dka" in data[9]["ideal"]:
        data[9]["ideal"] = "The diagnosis is type two diabetes with possible DKA."
        total_fixed += 1

    save("multistep-spellcaps.json", data)
    print(f"  multistep-spellcaps: {total_fixed} fixed")

    # ── 2. multistep-quote-corr: fix @-symbol placement + pronoun ──
    data = load("multistep-quote-corr.json")
    qc_fixes = 0

    # #3 (idx 2): pronoun resolution — "they're" should stay, not become "the mockups are"
    if "the mockups are pending review" in data[2]["ideal"]:
        data[2]["ideal"] = 'Message the @design team. Tell them "they\'re pending review".'
        qc_fixes += 1

    # #6 (idx 5): @ before Chen, not before Dr.
    if "@Dr. Chen" in data[5]["ideal"]:
        data[5]["ideal"] = 'Text Dr. @Chen about the appointment. Say "the results need follow-up".'
        qc_fixes += 1

    # #7 (idx 6): @ before coordinator, not before venue
    if "@venue coordinator" in data[6]["ideal"]:
        data[6]["ideal"] = 'Message the venue @coordinator. Tell her "we need the room until eleven".'
        qc_fixes += 1

    # #9 (idx 8): @ before Rosa, not before Aunt
    if "@Aunt Rosa" in data[8]["ideal"]:
        data[8]["ideal"] = 'Tell Aunt @Rosa about the flight. Say "I land at two in the afternoon".'
        qc_fixes += 1

    # #10 (idx 9): @ before manager, not before band
    if "@band manager" in data[9]["ideal"]:
        data[9]["ideal"] = 'Message the band @manager. Tell him "we\'re still deciding".'
        qc_fixes += 1

    save("multistep-quote-corr.json", data)
    total_fixed += qc_fixes
    print(f"  multistep-quote-corr: {qc_fixes} fixed")

    # ── 3. multistep-corr-format: fix scope issues ──
    data = load("multistep-corr-format.json")
    cf_fixes = 0

    # #4 (idx 3): "Scratch that, get the red one" = full replacement. Also caps.
    if "Buy the RED one" in data[3]["ideal"]:
        data[3]["ideal"] = "GET THE RED ONE."
        cf_fixes += 1

    # #9 (idx 8): consistent "scratch that" handling — full discard + caps on brand
    # Actually this one is fine as partial since "he said Gibson" provides replacement context
    # But let's keep it consistent with #4. The reviewer flagged inconsistency.
    # "Scratch that, he said Gibson" — the context "he said" makes it clear it's just the brand.
    # Leave as-is for now — it's a valid partial correction.

    # #10 (idx 9): "Make the day lowercase" should only lowercase the day, not everything
    if data[9]["ideal"] == "the team is practicing on thursday morning.":
        data[9]["ideal"] = "The team is practicing on thursday morning."
        cf_fixes += 1

    save("multistep-corr-format.json", data)
    total_fixed += cf_fixes
    print(f"  multistep-corr-format: {cf_fixes} fixed")

    # ── 4. multistep-complex: fix Golden Rule + emphasis + self-correction ──
    data = load("multistep-complex.json")
    cx_fixes = 0

    # #1 (idx 0): "Emphasize fifteen percent" → both words caps
    if 'FIFTEEN percent' in data[0]["ideal"]:
        data[0]["ideal"] = data[0]["ideal"].replace("FIFTEEN percent", "FIFTEEN PERCENT")
        cx_fixes += 1

    # #2 (idx 1): chef kiss emoji → use 🤌 (single emoji)
    if "👨\u200d🍳😘" in data[1]["ideal"]:
        data[1]["ideal"] = data[1]["ideal"].replace("👨\u200d🍳😘", "🤌")
        cx_fixes += 1

    # #4 (idx 3): self-correction echo + messy output — rewrite ideal
    if 'Actually, keep it' in data[3]["ideal"]:
        data[3]["ideal"] = 'Tell the @drummer the setlist needs changes. Keep Wonderwall but move it LAST. 🥁 The opener is now SMELLS LIKE TEEN SPIRIT.'
        cx_fixes += 1

    # #8 (idx 7): "florist" not in input — change input to include "florist"
    if "@florist" in data[7]["ideal"] and "florist" not in data[7]["input"].split("At symbol")[0]:
        data[7]["input"] = "ah Tell the florist the Peony arrangements are canceled. Spell P-E-O-N-Y, capitalize normally. At symbol before florist. Emphasize canceled. Say quote we switched to succulents end quote. Actually, quote we switched to orchids end quote."
        data[7]["ideal"] = 'Tell the @florist the Peony arrangements are CANCELED. Say "we switched to orchids".'
        cx_fixes += 1

    # #9 (idx 8): "assistant" not in input — change input to include "assistant coach"
    if "@assistant coach" in data[8]["ideal"] and "assistant" not in data[8]["input"].split("At symbol")[0]:
        data[8]["input"] = "um Message the assistant coach that the scrimmage is off. At symbol before assistant. The Gatorade order didn't come through. Spell G-A-T-O-R-A-D-E, capitalize normally. Emphasize off. Actually, the Powerade order. Say quote practice indoors instead end quote."
        cx_fixes += 1

    # #10 (idx 9): "conservator" not in input — change input to include "conservator"
    if "@conservator" in data[9]["ideal"] and "conservator" not in data[9]["input"].split("At symbol")[0]:
        data[9]["input"] = "ah Email the conservator the Klimt reproduction is damaged. Spell K-L-I-M-T, capitalize normally. At symbol before conservator. Emphasize damaged. Say quote insurance claim pending end quote. Actually, quote insurance claim filed end quote. The valuation is wrong too."
        cx_fixes += 1

    save("multistep-complex.json", data)
    total_fixed += cx_fixes
    print(f"  multistep-complex: {cx_fixes} fixed")

    # ── 5. spell-casual: fix trigger phrase + non-error ──
    data = load("spell-casual.json")
    sc_fixes = 0

    # #5 (idx 4): "for short" is wrong trigger — change to "by the way"
    if "for short" in data[4]["input"]:
        data[4]["input"] = "The game is called Fortnight. That's F-O-R-T-N-I-T-E by the way."
        sc_fixes += 1

    # #6 (idx 5): Calcutta is not an ASR error — use a plausible mishearing
    if "Calcutta" in data[5]["input"]:
        data[5] = {
            "input": "The new employee is from Colcata. It's actually K-O-L-K-A-T-A by the way.",
            "ideal": "The new employee is from Kolkata."
        }
        sc_fixes += 1

    save("spell-casual.json", data)
    total_fixed += sc_fixes
    print(f"  spell-casual: {sc_fixes} fixed")

    # ── 6. spell-alt-phrase: DELETE entirely (all phonetic breakdowns) ──
    alt_path = RAW / "spell-alt-phrase.json"
    if alt_path.exists():
        data = load("spell-alt-phrase.json")
        total_deleted = len(data)
        alt_path.unlink()
        print(f"  spell-alt-phrase: DELETED ({total_deleted} bad examples)")

    # ── 7. spell-compound-scope: fix CamelCase splitting ──
    data = load("spell-compound-scope.json")
    cs_fixes = 0

    # #1: HillTop → Hilltop (dictionary word, keep fused but lowercase t)
    # Actually "Hilltop" is already fine as a dictionary word. Reviewer wanted "Hill Top"
    # but "hilltop" IS one word in English. Let's keep "Hilltop" — it's correct.
    # No fix needed.

    # #2: BoneBroth → Bone Broth (not a brand)
    if "BoneBroth" in data[1]["ideal"]:
        data[1]["ideal"] = "My grandmother swears by Bone Broth for joint health."
        cs_fixes += 1

    # #3: MapleStreet → Maple Street (not a brand)
    if "MapleStreet" in data[2]["ideal"]:
        data[2]["ideal"] = "The bakery on Maple Street makes amazing croissants."
        cs_fixes += 1

    # #4: ToothBrush → toothbrush (common noun, dictionary word)
    if "ToothBrush" in data[3]["ideal"]:
        data[3]["ideal"] = "I need to buy a new toothbrush for my trip."
        cs_fixes += 1

    # #5: DustBowl → Dust Bowl (historical proper noun, two words)
    if "DustBowl" in data[4]["ideal"]:
        data[4]["ideal"] = "The documentary covers the Dust Bowl era farmers."
        cs_fixes += 1

    save("spell-compound-scope.json", data)
    total_fixed += cs_fixes
    print(f"  spell-compound-scope: {cs_fixes} fixed")

    # ── 8. emphasis-caps: fix sentence merging + dropped article ──
    data = load("emphasis-caps.json")
    ec_fixes = 0

    # #5 (idx 4): sentence merging — keep period, capitalize "We"
    if "CONSUMABLES ready, we wiped" in data[4]["ideal"]:
        data[4]["ideal"] = "For the raid tonight everyone needs to be on voice chat and have their CONSUMABLES ready. We wiped last time because people ran out of mana potions."
        ec_fixes += 1

    # #6 (idx 5): sentence merging — keep period
    if "RUNWAY, that's" in data[5]["ideal"]:
        data[5]["ideal"] = "The quarterly numbers look solid but I'm more worried about cash RUNWAY. That's what keeps me up at night."
        ec_fixes += 1

    # #10 (idx 9): dropped article "the" — add it back
    if "EMAIL VERIFICATION STEP, that's" in data[9]["ideal"]:
        data[9]["ideal"] = "The onboarding flow is functional but users are dropping off. The EMAIL VERIFICATION STEP, that's our biggest leak."
        ec_fixes += 1

    save("emphasis-caps.json", data)
    total_fixed += ec_fixes
    print(f"  emphasis-caps: {ec_fixes} fixed")

    # ── Summary ──
    print(f"\nTotal: {total_fixed} fixed, {total_deleted} deleted")
    print(f"\nNote: spell-alt-phrase.json was deleted. Need to regenerate with realistic ASR errors.")


if __name__ == "__main__":
    main()
