#!/usr/bin/env python3
"""
TrophicTrace — Fish Species Presence by Location
=================================================
Builds a state + lat/lon based species presence lookup using:
  1. Pre-fetched GBIF hotspot data (14 hotspots, live GBIF data)
  2. Live GBIF API queries for broader state-level coverage
  3. Known species ranges from fisheries biology literature

Output: backend/species_presence.json
"""

import json
import math
import time
import urllib.request
import urllib.error
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
OUTPUT = BACKEND_DIR / "species_presence.json"

# 8 target species with GBIF taxon keys and known range info
FISH_SPECIES = [
    {
        "common_name": "Largemouth Bass",
        "scientific_name": "Micropterus salmoides",
        "gbif_key": 2382397,
        # Native east of Rockies, introduced widely
        "native_states": ["AL","AR","CT","DE","FL","GA","IA","IL","IN","KS","KY",
                          "LA","MA","MD","ME","MI","MN","MO","MS","NC","NE","NH",
                          "NJ","NY","OH","OK","PA","RI","SC","SD","TN","TX","VA",
                          "VT","WI","WV"],
        "introduced_states": ["AZ","CA","CO","HI","ID","MT","NM","NV","OR","UT","WA","WY"],
    },
    {
        "common_name": "Channel Catfish",
        "scientific_name": "Ictalurus punctatus",
        "gbif_key": 2341144,
        "native_states": ["AL","AR","FL","GA","IA","IL","IN","KS","KY","LA","MI",
                          "MN","MO","MS","NC","NE","NJ","NY","OH","OK","PA","SC",
                          "SD","TN","TX","VA","WI","WV"],
        "introduced_states": ["AZ","CA","CO","CT","DE","ID","MA","MD","MT","NH",
                              "NM","NV","OR","RI","UT","VT","WA","WY"],
    },
    {
        "common_name": "Striped Bass",
        "scientific_name": "Morone saxatilis",
        "gbif_key": 2385645,
        # Anadromous — Atlantic coast + major reservoirs
        "native_states": ["CT","DE","FL","GA","MA","MD","ME","NC","NH","NJ","NY",
                          "PA","RI","SC","VA"],
        "introduced_states": ["AL","AR","CA","CO","IL","KS","KY","LA","MI","MO",
                              "MS","NE","NM","NV","OH","OK","OR","TN","TX","WA"],
    },
    {
        "common_name": "Brook Trout",
        "scientific_name": "Salvelinus fontinalis",
        "gbif_key": 5202559,
        # Cold-water — Appalachians, Great Lakes, New England
        "native_states": ["CT","GA","IA","IL","IN","KY","MA","MD","ME","MI","MN",
                          "NC","NH","NJ","NY","OH","PA","RI","SC","TN","VA","VT",
                          "WI","WV"],
        "introduced_states": ["AZ","CA","CO","ID","MT","NM","NV","OR","SD","UT",
                              "WA","WY"],
    },
    {
        "common_name": "Yellow Perch",
        "scientific_name": "Perca flavescens",
        "gbif_key": 2382763,
        "native_states": ["CT","IA","IL","IN","KS","MA","MD","ME","MI","MN","MO",
                          "MT","NC","NE","NH","NJ","NY","OH","PA","RI","SC","SD",
                          "VA","VT","WI","WV"],
        "introduced_states": ["AL","AZ","CA","CO","GA","ID","NM","NV","OR","TX",
                              "UT","WA","WY"],
    },
    {
        "common_name": "Common Carp",
        "scientific_name": "Cyprinus carpio",
        "gbif_key": 4286952,
        # Introduced everywhere in continental US
        "native_states": [],
        "introduced_states": ["AL","AR","AZ","CA","CO","CT","DE","FL","GA","IA",
                              "ID","IL","IN","KS","KY","LA","MA","MD","ME","MI",
                              "MN","MO","MS","MT","NC","NE","NH","NJ","NM","NV",
                              "NY","OH","OK","OR","PA","RI","SC","SD","TN","TX",
                              "UT","VA","VT","WA","WI","WV","WY"],
    },
    {
        "common_name": "Smallmouth Bass",
        "scientific_name": "Micropterus dolomieu",
        "gbif_key": 2382407,
        "native_states": ["AL","AR","GA","IA","IL","IN","KS","KY","MI","MN","MO",
                          "MS","NC","NE","NY","OH","OK","PA","TN","VA","WI","WV"],
        "introduced_states": ["AZ","CA","CO","CT","DE","ID","MA","MD","ME","MT",
                              "NH","NJ","NM","NV","OR","RI","SC","SD","TX","UT",
                              "VT","WA","WY"],
    },
    {
        "common_name": "White Sucker",
        "scientific_name": "Catostomus commersonii",
        "gbif_key": 2365034,
        "native_states": ["AL","AR","CT","GA","IA","IL","IN","KS","KY","MA","MD",
                          "ME","MI","MN","MO","MS","MT","NC","NE","NH","NJ","NM",
                          "NY","OH","OK","PA","RI","SC","SD","TN","VA","VT","WI",
                          "WV","WY"],
        "introduced_states": ["CO","ID","NV","OR","UT","WA"],
    },
]

# State abbreviation to full name mapping
STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}


def fetch_gbif_count(species_key, state_name):
    """Get GBIF occurrence count for a species in a US state."""
    url = (f"https://api.gbif.org/v1/occurrence/search"
           f"?taxonKey={species_key}&country=US"
           f"&stateProvince={urllib.parse.quote(state_name)}"
           f"&limit=0")
    req = urllib.request.Request(url, headers={"User-Agent": "TrophicTrace/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("count", 0)
    except Exception:
        return None


def build_species_presence():
    """Build species presence lookup combining known ranges + GBIF verification."""
    print("=" * 60)
    print("TrophicTrace — Fish Species Presence Builder")
    print("=" * 60)

    # Build state-level presence map
    state_species = {}  # {state_abbr: [species_name, ...]}
    for abbr in STATE_ABBR_TO_NAME:
        state_species[abbr] = []

    # Start with known ranges (authoritative, fast)
    for sp in FISH_SPECIES:
        for state in sp["native_states"] + sp["introduced_states"]:
            if state in state_species:
                state_species[state].append(sp["common_name"])

    # Load hotspot data to enrich with GBIF occurrence counts
    hotspot_path = PROJECT_DIR / "new_data" / "real_data" / "fish_species_presence_real.json"
    hotspot_species = {}
    if hotspot_path.exists():
        with open(hotspot_path) as f:
            hs_data = json.load(f)
        for hs in hs_data.get("fish_presence_data", []):
            hotspot_species[hs["hotspot_id"]] = {
                sp["common_name"]: sp["occurrence_count"]
                for sp in hs.get("species", [])
            }
        print(f"  Loaded GBIF data for {len(hotspot_species)} hotspots")

    # Verify a sample of state-species combinations via GBIF
    print("\n  Verifying sample state-species combos via GBIF...")
    verified = 0
    sample_states = ["NC", "MI", "AL", "PA", "NY", "CO", "NH", "NJ", "VT", "WV", "FL", "CA", "TX"]
    for sp in FISH_SPECIES:
        for state_abbr in sample_states:
            state_name = STATE_ABBR_TO_NAME.get(state_abbr, "")
            count = fetch_gbif_count(sp["gbif_key"], state_name)
            if count is not None:
                verified += 1
                if count > 0 and sp["common_name"] not in state_species.get(state_abbr, []):
                    # GBIF says present but not in our known ranges — add it
                    state_species.setdefault(state_abbr, []).append(sp["common_name"])
                elif count == 0 and sp["common_name"] in state_species.get(state_abbr, []):
                    # Known range says present but GBIF has 0 records — keep it
                    # (GBIF may just lack coverage, known ranges are more authoritative)
                    pass
            time.sleep(0.15)

        print(f"    {sp['common_name']}: verified {len(sample_states)} states")

    print(f"\n  Total GBIF verifications: {verified}")

    # Build output
    result = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": [
                "Known species ranges (FishBase, state wildlife agencies)",
                "GBIF Occurrence API (live verification)",
                "Pre-fetched hotspot GBIF data",
            ],
            "n_species": len(FISH_SPECIES),
            "n_states": len(state_species),
        },
        "species_info": [
            {
                "common_name": sp["common_name"],
                "scientific_name": sp["scientific_name"],
                "gbif_key": sp["gbif_key"],
                "n_states_present": len(sp["native_states"]) + len(sp["introduced_states"]),
            }
            for sp in FISH_SPECIES
        ],
        "state_species": state_species,
        "hotspot_species": hotspot_species,
    }

    with open(OUTPUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUTPUT}")

    # Summary
    for sp in FISH_SPECIES:
        total_states = sum(1 for states in state_species.values() if sp["common_name"] in states)
        print(f"  {sp['common_name']:20s} present in {total_states} states")

    return result


if __name__ == "__main__":
    build_species_presence()
