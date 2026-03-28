# TrophicTrace — Neural PFAS Bioaccumulation Prediction & Fish Safety Advisory System

## Hackathon Project Specification (19 Hours, 4 People)

---

## Abstract

PFAS ("forever chemicals") contaminate the drinking water of over 100 million Americans, but the far more dangerous and understudied threat is **food-web bioaccumulation**: PFAS discharged into waterways concentrate up the food chain by 1,000–10,000x, meaning water deemed "safe" produces dangerous tissue-level exposures in fish consumed by humans. A 2023 EWG study found that eating **a single serving of freshwater fish** delivers PFAS exposure equivalent to drinking contaminated water for an entire month. The EPA's 2024 drinking water regulation explicitly states that PFAS regulation will "prevent thousands of deaths," yet fish consumption advisories remain state-level, years out of date, and blind to the environmental justice communities — subsistence fishers, disproportionately low-income, Indigenous, and Black populations — who face 3–5x higher exposure than recreational anglers.

**TrophicTrace** is the first system to predict PFAS contamination levels across an entire aquatic food web using a **physics-informed heterogeneous graph neural network (GNN)** trained on environmental, ecological, and chemical data. Rather than relying on sparse, expensive field sampling, TrophicTrace learns the relationship between industrial discharge, hydrologic transport, trophic transfer, and tissue-level bioaccumulation — then generalizes to predict contamination for unsampled species, locations, and scenarios. The system produces an **interactive geographic heatmap** showing predicted fish tissue contamination across a watershed, with species-specific consumption advisories and interpretable explanations of the contamination drivers at each point.

The entire pipeline runs on an **ASUS supercomputer**: (1) training the heterogeneous GNN with physics-informed constraints requires GPU-accelerated constrained optimization over large multi-typed graphs, and (2) state environmental agencies and tribal nations handle legally privileged pre-enforcement data with sovereignty requirements that prohibit uploading to external cloud servers.

**The cost of inaction is staggering.** A 2025 Systemiq report found that toxic chemicals in food systems impose up to **$3 trillion annually** in preventable health costs. PFAS remediation costs $0.9–65 million per kilogram. Prediction and prevention is the only economically viable path. TrophicTrace gives every state agency and tribal environmental office the analytical capacity that currently requires a team of environmental toxicologists — delivered through a single interactive map.

**Key sourced statistics for judges:**
- 1 freshwater fish serving = 1 month of PFAS water exposure (EWG 2023)
- PFAS bioaccumulate 600x+ in downstream biota vs. water (USGS 2025)
- EPA 2024: PFAS regulation will "prevent thousands of deaths"
- $3 trillion/year global cost of toxic chemicals in food (Systemiq 2025)
- $47–75 billion/year US healthcare costs from PFAS (NRDC 2025)
- $0.9–65M per kg PFAS remediation — prevention is the only viable path (Wisconsin DNR 2024)
- Fish advisories are years out of date — MN updated in 2026 based on multi-year review cycles
- Subsistence fishers face 3–5x the exposure of recreational anglers (EPA 2024/2025)
- Tribal data sovereignty requires on-premises AI (2026 Indigenous Data Sovereignty Summit)

---

## The Product: What TrophicTrace Is, In Detail

TrophicTrace has three layers: a **trained AI model** that predicts contamination, a **visualization** that makes the predictions explorable, and an optional **chatbot** that lets non-technical users ask questions in natural language.

---

### Layer 1: The Physics-Informed Heterogeneous GNN

#### What problem it solves

Today, if a state agency wants to know whether largemouth bass in a specific lake are safe to eat, they must physically collect fish, ship tissue samples to a lab, wait weeks for PFAS analysis, and repeat this across dozens of species and locations. This costs tens of thousands of dollars per watershed and produces a snapshot that's immediately out of date. Most watersheds have never been sampled at all.

TrophicTrace replaces this with a **learned model** that takes in readily available data — what facilities discharge where, how the river network flows, what species live where and eat what, and basic water chemistry — and predicts tissue-level PFAS concentrations everywhere in the watershed, for every species, simultaneously. No lab work. No sampling. Just data and physics.

#### Why a GNN is the right architecture

The data is inherently a graph. A watershed is a network of interconnected entities:

- **Facilities** discharge PFAS into specific **river segments**
- **River segments** are connected by flow direction (upstream→downstream), carrying dissolved PFAS
- **Fish species** inhabit specific segments and are connected by **trophic (feeding) relationships** — bluegill eat invertebrates, largemouth bass eat bluegill
- **Human populations** near the river consume specific species at specific rates

PFAS contamination propagates through this entire network: a factory discharges GenX → it flows downstream → algae absorb it → invertebrates eat algae and concentrate it 10x → bluegill eat invertebrates and concentrate it 100x → largemouth bass eat bluegill and concentrate it 1000x → a subsistence fisher eats bass three times a week. Each step is a message being passed along an edge of the graph. A GNN learns exactly these message-passing dynamics.

#### Why "physics-informed"

Pure data-driven learning would require massive labeled datasets (thousands of fish tissue samples across many watersheds) that don't exist. Instead, we encode known environmental chemistry as constraints in the training loss. These act as inductive biases that dramatically reduce the amount of training data needed:

1. **Trophic monotonicity:** For bioaccumulative PFAS, concentrations MUST increase at higher trophic levels. If the model predicts that a bass (predator) has lower PFAS than a bluegill (prey it eats), that's physically impossible. We penalize this in the loss.

2. **BCF bounds:** The bioconcentration factor (tissue concentration ÷ water concentration) for a given species and chemical is bounded by the chemical's octanol-water partition coefficient — a known physical property. If the model's implied BCF falls outside published ranges, we penalize it.

3. **Mass conservation:** At river confluences, PFAS mass in = PFAS mass out (minus sediment deposition). The model must respect water-balance physics.

These constraints mean the model can learn from far fewer labeled examples because the physics rules out most wrong answers before training even begins.

#### Architecture specifics

**Node types (4):**

| Node Type | Count per Graph | Attributes | Meaning |
|-----------|----------------|------------|---------|
| Facility | 3–8 | Discharge volume, congener emission vector (6-dim: PFOA, PFOS, GenX, PFHxS, PFNA, PFDA), lat/lng | An industrial discharge source |
| Waterbody | 20–50 | Flow rate (m³/s), pH, temperature (°C), dissolved organic carbon (mg/L), segment length (km), lat/lng | A river/stream segment |
| Species | 30–80 (species × locations) | Trophic level, lipid content (%), body mass (g), metabolic rate, species one-hot | A fish species at a specific location |
| Population | 2–4 | Consumption rate vector (servings/month by species), population size, subsistence flag, median income | A demographic group |

**Edge types (5):**

| Edge Type | Meaning | Typical Count |
|-----------|---------|--------------|
| Facility → Waterbody | "discharges into" | 3–8 |
| Waterbody → Waterbody | "flows downstream to" | 20–50 |
| Waterbody → Species | "is habitat of" | 50–150 |
| Species → Species | "is eaten by" (trophic) | 80–200 |
| Species → Population | "is consumed by" | 30–80 |

**Model:**

- **Node-type-specific encoders:** Each node type has a 2-layer MLP mapping raw features → shared hidden dimension (64 or 128).
- **3 layers of HeteroConv** (PyTorch Geometric): each edge type gets its own convolutional operator. The trophic (species→species) edges use `GATConv` because attention weights directly give us interpretability — the model learns which prey species contribute most to each predator's contamination. All other edge types use `SAGEConv`.
- **Prediction head:** 2-layer MLP on species node embeddings → scalar tissue concentration (ng/g).
- **~50K–200K parameters** total. This is a small model — physics constraints do the heavy lifting.

**Training:**

- 8 synthetic watershed graphs for training, 2 for validation
- Labels generated by a forward physical simulation (discharge → dilution → BCF → BMF → tissue concentration + 10–20% noise) using published EPA ECOTOX values
- Loss = MSE + 0.1 × monotonicity penalty + 0.1 × BCF bound penalty
- Adam optimizer, lr=1e-3, 300–500 epochs
- Trains in <30 min on a single ASUS GPU

**Inference:** For every species-at-location node in the Cape Fear River demo graph, the model outputs a predicted tissue concentration. GATConv attention weights and gradient-based feature attribution provide interpretability data.

---

### Layer 2: The Interactive Visualization

This is what the judges see first and what makes the project memorable. It must be beautiful, intuitive, and tell a story about environmental injustice through data. Here is exactly what every screen looks like.

---

#### Screen 1: The Main Map View (Full Screen)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ┌─ Title Bar ────────────────────────────────────────────────────────┐  │
│  │  🐟 TrophicTrace        Cape Fear River, NC      [EJ Overlay ◉]  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│                                                                          │
│               ⚠ Chemours                                                │
│               Fayetteville                                               │
│               Works                                                      │
│                  \                                                        │
│                   ·~·~·~~~~river segment, green/thin~~~~·~·~             │
│                             \                                            │
│                              ·~·~ORANGE, thicker~·~·~·~                 │
│                                                   \                      │
│                          [soft red glow]  ·~~RED, THICK~~·~·            │
│                                                          \               │
│                                    ⚠ Fayetteville        ·~~·~~·       │
│                                      WWTP                      \         │
│                                                                  ·~~·   │
│                                                                   ↓     │
│                                                              to ocean   │
│                                                                          │
│                                                                          │
│  ┌─ Legend ─────────────────────────────────────────────┐                │
│  │  Fish Tissue PFAS (ng/g):                            │                │
│  │  ██ <5 Safe   ██ 5–20 Limited   ██ >20 Unsafe       │                │
│  └──────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────┘
```

**Base map:** Dark-themed Mapbox style (e.g., `mapbox://styles/mapbox/dark-v11`). The darkness makes data layers visually pop. Centered on the Cape Fear River watershed in North Carolina — roughly Fayetteville area (lat ~34.5–35.3, lng ~-79.0 to -77.9, zoom ~9–10).

**River network layer:** The river is drawn as GeoJSON `LineString` features on top of the base map. Each segment is styled with two properties that encode contamination data:

- **Color:** Continuous gradient based on predicted water-column PFAS concentration at that segment. Low contamination = blue-green (#22d3ee). Medium = amber (#f59e0b). High = red (#ef4444). Use Mapbox's `line-color` with `interpolate` expression on the `water_pfas_ppt` property.
- **Width:** Segments get thicker downstream of discharge points, representing increasing contamination load. Ranges from 2px (clean headwaters) to 6px (heavily contaminated segments). Use Mapbox's `line-width` with `interpolate` expression.
- **Glow effect:** Each river line is drawn twice — once as the colored line, and once as a wider (3x), more transparent version underneath, creating a soft neon glow effect. This makes the river visually striking on the dark background.

**Heatmap layer:** A soft, translucent radial glow around each river segment, colored by the MAXIMUM predicted fish tissue concentration at that segment:

- **Green (#22c55e / 10% opacity):** All species safe. Barely visible glow, 50px radius.
- **Amber (#eab308 / 20% opacity):** Some species limited. Moderate glow, 80px radius.
- **Red (#ef4444 / 30% opacity):** Some species unsafe. Intense glow, 120px radius.

Implementation: Mapbox heatmap layer using segment centroids as points, with `heatmap-weight` driven by max tissue concentration, and `heatmap-color` using the green→amber→red ramp. Alternatively, custom Canvas overlay with radial gradients at each point.

The overall visual effect: a mostly calm, blue-green watershed that transitions into angry red/orange downstream of the Chemours facility. The story of contamination spreading is told instantly by the colors.

**Facility markers:** White or yellow factory/hazard icons placed at discharge facility locations. Each has a subtle CSS pulse animation (concentric rings expanding and fading). On hover, show a small tooltip with facility name and primary discharge info. These should be eye-catching — they're the villains of the story.

**EJ overlay toggle:** A toggle switch in the title bar. When activated, it shows semi-transparent polygon boundaries for census tracts with high subsistence fishing rates, filled with a purple/violet tint at ~15% opacity and a dashed border. Inside each polygon, a small label shows "Subsistence fishing: 18.5% | Median income: $31,200." This layer makes the environmental justice dimension literally visible on the map.

**Legend:** Bottom-left corner, compact, semi-transparent dark card. Shows the color ramp with thresholds (< 5, 5–20, > 20 ng/g) and a brief label. Should not compete with the map for attention.

---

#### Screen 2: The Hover Tooltip

When the cursor hovers over any point on the river or heatmap, a tooltip appears anchored near the cursor:

```
┌────────────────────────────────────────────────────┐
│                                                    │
│  📍 Cape Fear River — Fayetteville Reach           │
│  Water PFAS: 120.5 ppt                             │
│                                                    │
│  ─────────────────────────────────────────────      │
│                                                    │
│  ● Largemouth Bass              48.3 ng/g          │
│    ⚠ Max 1 serving/month                           │
│                                     [More Details] │
│                                                    │
│  ● Striped Bass                 38.9 ng/g          │
│    ⚠ Max 1 serving/month                           │
│                                     [More Details] │
│                                                    │
│  ● Channel Catfish              14.7 ng/g          │
│    ⚠ Max 3 servings/month                          │
│                                     [More Details] │
│                                                    │
│  ● Bluegill                      3.1 ng/g          │
│    ✓ Safe for regular consumption                  │
│                                                    │
│  ─────────────────────────────────────────────      │
│                                                    │
│  ┌──────────────────────────────────────────┐      │
│  │ ⚠ EJ ALERT: Subsistence fishers here    │      │
│  │ face 3.2× the exposure of recreational  │      │
│  │ anglers (Fayetteville SE, median income  │      │
│  │ $31,200, 18.5% subsistence fishing)     │      │
│  └──────────────────────────────────────────┘      │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Design specs:**

- **Background:** Dark card (rgba(15, 15, 25, 0.92)) with `backdrop-filter: blur(12px)` and a 1px border in rgba(255,255,255,0.08). Rounded corners (12px). Subtle box-shadow.
- **Typography:** System sans-serif or Inter. Location name in 14px semibold white. "Water PFAS" in 12px gray (#9ca3af). Species names in 13px medium white. Numbers in 13px monospace or tabular-nums.
- **Safety dots:** Small CSS circles (8px diameter) next to each species name:
  - Red (#ef4444) for unsafe (>20 ng/g)
  - Amber (#eab308) for limited (5–20 ng/g)
  - Green (#22c55e) for safe (<5 ng/g)
- **Species sorting:** Listed worst-first (highest tissue concentration at top). This puts the most dangerous information first.
- **"More Details" link:** Subtle text link in blue-gray (#60a5fa), right-aligned. Clicking opens the interpretability panel for that species.
- **Advisory text:** One-line recommendation per species. Uses ⚠ for limited/unsafe, ✓ for safe.
- **EJ alert box:** Only appears when the hovered segment overlaps or is near an EJ demographic zone. Has a distinct background (rgba(139, 92, 246, 0.15) — purple tint) with a left border accent (3px solid #8b5cf6). The content shows the exposure multiplier, community name, and key demographic stats.
- **Positioning:** Tooltip appears offset from cursor (20px right, 10px below). If it would overflow the viewport edge, flip to the other side. Fades in with 150ms CSS transition. When cursor moves to a different segment, the content crossfades (opacity out/in, 100ms).
- **Width:** Fixed at 340px. Height varies with number of species.

---

#### Screen 3: The Interpretability Panel ("More Details")

Clicking "More Details" on any species slides in a panel from the right edge:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                    ┌────────────────┤
│                                                    │   ✕            │
│                                                    │                │
│      (map visible but dimmed                       │  LARGEMOUTH    │
│       with dark scrim overlay,                     │  BASS          │
│       ~40% opacity black)                          │  Cape Fear R.  │
│                                                    │  — Fayetteville│
│                                                    │                │
│                                                    │ ┌────────────┐ │
│                                                    │ │            │ │
│                                                    │ │   48.3     │ │
│                                                    │ │   ng/g     │ │
│                                                    │ │            │ │
│                                                    │ │  EPA limit: │ │
│                                                    │ │  20 ng/g   │ │
│                                                    │ │            │ │
│                                                    │ │  2.4× OVER │ │
│                                                    │ │   (RED)    │ │
│                                                    │ └────────────┘ │
│                                                    │                │
│                                                    │ WHY IS THIS    │
│                                                    │ FISH UNSAFE?   │
│                                                    │                │
│                                                    │ Chemours    ████████████  62% │
│                                                    │ Trophic lvl █████  19%        │
│                                                    │ Water temp  ██  8%            │
│                                                    │ Lipid cont  ██  6%            │
│                                                    │ Low flow    █  5%             │
│                                                    │                │
│                                                    │ ────────────── │
│                                                    │                │
│                                                    │ CONTAMINATION  │
│                                                    │ PATHWAY        │
│                                                    │                │
│                                                    │ [Chemours]     │
│                                                    │  450 ppt       │
│                                                    │     │ ÷3.7     │
│                                                    │     ▼ dilution │
│                                                    │ [River Water]  │
│                                                    │  120.5 ppt     │
│                                                    │     │ ×400     │
│                                                    │     ▼ BCF      │
│                                                    │ [Fish Tissue]  │
│                                                    │  48.3 ng/g     │
│                                                    │     │          │
│                                                    │     ▼          │
│                                                    │ [Human Dose]   │
│                                                    │                │
│                                                    │ ────────────── │
│                                                    │                │
│                                                    │ WHO IS AT RISK?│
│                                                    │                │
│                                                    │ Recreational   │
│                                                    │ ████░░░░░░ 0.8×│
│                                                    │                │
│                                                    │ Subsistence    │
│                                                    │ ██████████████ │
│                                                    │ ████ 2.6× !!  │
│                                                    │        ↑       │
│                                                    │    EPA limit   │
│                                                    │                │
│                                                    └────────────────┤
└──────────────────────────────────────────────────────────────────────┘
```

**Panel specs:**

- **Width:** 420px. Slides in from the right with a 300ms ease-out CSS transition.
- **Background:** Solid dark (#0f0f19) or matching the tooltip style (dark with subtle blur). Full viewport height, scrollable if content overflows.
- **Close button:** Top-right ✕, clicking slides panel out and removes the map scrim.
- **Scrim:** When panel is open, the map area gets a dark overlay (rgba(0,0,0,0.4)) to focus attention on the panel.

**Section A — Header & Verdict:**

- Species common name: 22px bold white
- Location: 14px gray
- **Big number display:** The predicted tissue concentration in a large (48px) bold font. Colored red/amber/green by safety status. This is the hero element — it should be the first thing the eye lands on.
- Below it: "EPA reference dose: 20 ng/g" in 12px gray, and a multiplier badge: "2.4× OVER" in a red pill/badge, or "0.3× UNDER" in a green pill/badge.
- Recommended servings: "Maximum 1 serving per month" in 14px with ⚠ icon.

**Section B — "Why Is This Fish Unsafe?" (Contributing Factors):**

- Section header: 14px uppercase semibold gray, with a subtle horizontal rule above.
- **Horizontal bar chart:** 5 bars, sorted by contribution percentage (highest first). Each bar is:
  - Color-coded by factor type:
    - Source factors (discharge facilities): warm red/orange (#f97316)
    - Ecological factors (trophic level, food web): blue (#3b82f6)
    - Environmental factors (temperature, pH): teal (#14b8a6)
    - Biological factors (lipid content, body mass): purple (#a855f7)
    - Hydrologic factors (flow, dilution): gray (#6b7280)
  - Labeled with factor name on the left, percentage on the right
  - Bar width proportional to percentage
  - Built with pure CSS (flexbox width percentages) or D3 — either works
  - Each bar should have slightly rounded ends and a subtle hover highlight

**This data comes from the GNN's interpretability output:** gradient-based attribution of the species node prediction w.r.t. upstream node features, grouped and normalized by factor type.

**Section C — "Contamination Pathway" (Mini-Sankey):**

- A vertical flow diagram showing 4 nodes connected by lines:
  1. **[Chemours]** — 450 ppt (discharge concentration)
  2. **[River Water]** — 120.5 ppt (after dilution) — annotated with "÷3.7 dilution"
  3. **[Fish Tissue]** — 48.3 ng/g (after bioconcentration) — annotated with "×400 BCF"
  4. **[Human Dose]** — X ng/kg/day (consumption-adjusted)
- The connecting lines get thicker at each step to visually show concentration increasing
- Each node is a rounded rectangle with the concentration value inside
- Annotations (dilution factor, BCF) sit next to the connecting lines
- This is a simple custom SVG — NOT a full D3-sankey library. Just 4 boxes + 3 connecting paths with stroke-width varying.
- Color: nodes match the safety color of that stage (green→amber→red as concentration increases)

**Section D — "Who Is At Risk?" (Dose Comparison):**

- Two horizontal bars showing estimated dose for:
  1. **Recreational angler** (assumes 1 serving/month): bar extends to 0.8× the EPA reference dose (green, stays within safe zone)
  2. **Subsistence fisher** (assumes 8 servings/month): bar extends to 2.6× the EPA reference dose (red, extends far past the safe zone)
- A vertical dashed line marks the EPA reference dose threshold, labeled "EPA limit"
- The subsistence bar visually overshooting the line is the "aha moment" — it makes the EJ disparity impossible to miss
- Below the bars, a brief text note: "Subsistence fishers in Fayetteville SE (median income $31,200) face 3.2× the exposure of recreational anglers."

---

### Layer 3 (STRETCH GOAL): RAG-Based Fish Safety Chatbot

A chat interface where users type questions like "How much bass from the Cape Fear is safe to eat weekly?" and get grounded answers citing state advisories and GNN predictions. Uses RAG over state fish advisories + EPA data + GNN output.

**Build only if core GNN + visualization are integrated by hour 15.** Fallback: 4–5 pre-canned example Q&A pairs displayed as a static showcase.

---

## Data Contract: The JSON Schema

**This is the single most important coordination artifact.** Track A (GNN) produces this JSON. Track B/C (visualization) consumes it. Everyone builds against this schema from hour 0.

```json
{
  "watershed_name": "Cape Fear River, NC",
  "metadata": {
    "model_version": "trophictrace-gnn-v1",
    "training_epochs": 400,
    "training_device": "ASUS [GPU model]",
    "inference_timestamp": "2026-03-28T14:30:00Z"
  },
  "segments": [
    {
      "segment_id": "seg_001",
      "name": "Cape Fear River — Fayetteville Reach",
      "lat": 35.0527,
      "lng": -78.8784,
      "water_pfas_concentration_ppt": 120.5,
      "flow_rate_m3s": 45.2,
      "ph": 7.1,
      "temperature_c": 22.5,
      "species": [
        {
          "common_name": "Largemouth Bass",
          "scientific_name": "Micropterus salmoides",
          "trophic_level": 4.2,
          "lipid_content_pct": 5.8,
          "tissue_concentration_ng_g": 48.3,
          "safe_servings_per_month": 1,
          "safety_status": "unsafe",
          "contributing_factors": [
            { "factor": "Chemours Fayetteville Works discharge", "contribution_pct": 62.1, "type": "source" },
            { "factor": "High trophic level (4.2)", "contribution_pct": 18.7, "type": "ecological" },
            { "factor": "Elevated water temperature (24°C)", "contribution_pct": 8.4, "type": "environmental" },
            { "factor": "High lipid content (5.8%)", "contribution_pct": 6.2, "type": "biological" },
            { "factor": "Low flow dilution", "contribution_pct": 4.6, "type": "hydrologic" }
          ],
          "pathway": {
            "source_name": "Chemours Fayetteville Works",
            "discharge_ppt": 450,
            "water_concentration_ppt": 120.5,
            "dilution_factor": 3.7,
            "bcf_applied": 400,
            "tissue_concentration_ng_g": 48.3,
            "epa_reference_dose_ng_g": 20.0,
            "recreational_dose_ng_kg_day": 0.82,
            "subsistence_dose_ng_kg_day": 2.63
          }
        }
      ]
    }
  ],
  "facilities": [
    {
      "facility_id": "fac_001",
      "name": "Chemours Fayetteville Works",
      "lat": 34.9884,
      "lng": -78.8375,
      "pfas_discharge_ppt": 450,
      "congener_profile": {
        "GenX": 0.65, "PFOA": 0.15, "PFOS": 0.12,
        "PFHxS": 0.05, "PFNA": 0.02, "PFDA": 0.01
      }
    }
  ],
  "demographics": [
    {
      "area_name": "Fayetteville Southeast",
      "center_lat": 35.03,
      "center_lng": -78.85,
      "boundary_coords": [[-78.90, 35.05], [-78.80, 35.05], [-78.80, 35.00], [-78.90, 35.00]],
      "subsistence_fisher_pct": 18.5,
      "median_income": 31200,
      "majority_race": "Black",
      "exposure_multiplier_vs_recreational": 3.2,
      "population": 24500
    }
  ],
  "river_geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "properties": {
          "segment_id": "seg_001",
          "water_pfas_ppt": 120.5,
          "max_tissue_ng_g": 48.3,
          "safety_status": "unsafe"
        },
        "geometry": {
          "type": "LineString",
          "coordinates": [[-78.90, 35.06], [-78.88, 35.05], [-78.85, 35.04]]
        }
      }
    ]
  }
}
```

---

## ASUS Integration

1. **GNN Training (compute-critical).** Training involves backpropagation through 3 layers of heterogeneous message passing on graphs with ~10³–10⁴ nodes and ~10⁴ edges, with physics-informed loss terms requiring full-graph constraint evaluation each step. 300–500 epochs across 8–10 graphs. Laptop CPU = hours. ASUS GPU = <30 minutes. We show training logs/curves from the ASUS in the demo.

2. **On-premises inference.** Trained model runs inference on the Cape Fear River demo graph entirely on ASUS. Demonstrates full on-prem capability.

3. **Data sovereignty.** State environmental agencies handle pre-enforcement PFAS data that is legally privileged. Tribal nations have data sovereignty requirements (2026 Indigenous Data Sovereignty Summit). ASUS enables the full pipeline on sovereign infrastructure. Demo line: "This model was trained on [ASUS specs]. No data left this machine."

---

## Work Breakdown: 4 Parallel Technical Tracks

All 4 tracks are fully technical — all 4 people are writing code for 14+ hours. Slides are assembled from screenshots + talking points in the final 90 minutes. No dedicated presentation person.

The tracks are designed so that **zero coordination is needed for the first 12 hours**. Each track has its own clear deliverables, and the interfaces between tracks are defined by the JSON schema above and a shared GeoJSON file.

---

### TRACK A: GNN Model + ASUS Training Pipeline

**Owner:** Strongest ML/PyTorch person
**Tools:** Python 3.10+, PyTorch, PyTorch Geometric, NumPy, NetworkX, CUDA, matplotlib
**Independent until:** Hour 14 (delivers `demo_data.json` to Track B)
**Depends on:** Track D delivers Cape Fear parameters by hour 5

#### A1: Synthetic Data Generator (Hours 0–4)

Build `generate_data.py`.

**Function 1: `generate_watershed(n_facilities, n_segments, n_species, seed)`**
- Create facility nodes with random discharge (10–1000 ppt) and congener profiles (6-dim Dirichlet sample)
- Create waterbody nodes as a DAG (random tree with 1–3 tributaries merging into main stem). Each segment: flow rate 0.1–100 m³/s (increasing downstream as tributaries join), pH 6.5–8.5, temp 5–30°C, DOC 1–15 mg/L
- Connect facilities to nearest downstream segments
- Create species with: trophic level 2.0–4.5, lipid content 1–10%, body mass 10–5000g, metabolic rate from allometric scaling
- Place each species at 3–8 segments → creates species-at-location nodes
- Create trophic edges: lower trophic species → higher trophic species (with some randomness)
- Create 2–3 population groups with consumption rate vectors
- Return as PyTorch Geometric `HeteroData`

**Function 2: `simulate_forward(graph)`**
- Propagate discharge through river DAG: concentration = (upstream_conc × upstream_flow + discharge) / total_flow
- Water→species: tissue = water_concentration × BCF (BCF sampled from ranges based on lipid content, 100–5000)
- Species→species (trophic): apply BMF 2–15 based on trophic level difference
- Add 10–20% Gaussian noise
- Store as labels on species nodes

**Generate:** 8 training + 2 validation graphs (varied sizes). Also 1 Cape Fear demo graph using real parameters from Track D.

**Deliverable:** `generate_data.py`, 11 `.pt` files.

#### A2: GNN Architecture (Hours 4–7)

Build `model.py` with `TrophicTraceGNN`:
- 4 node-type encoders (2-layer MLPs → hidden_dim=64)
- 3 `HeteroConv` layers: `SAGEConv` on most edge types, `GATConv` on trophic edges (for attention-based interpretability)
- Prediction head: 2-layer MLP → scalar tissue concentration
- ReLU activations between layers

Build `loss.py` with `physics_informed_loss`:
- MSE on observed tissue concentrations
- Trophic monotonicity penalty: `relu(prey_concentration - predator_concentration).mean()`
- BCF bounds penalty: penalize when implied BCF falls outside [bcf_lower, bcf_upper]
- Total = MSE + α × monotonicity + β × BCF_bounds (α=β=0.1)

**Verify:** Forward pass runs without errors on a sample graph.

**Risk mitigation:** If PyG `HeteroConv` is problematic, implement manually with `torch.nn.Linear` + sparse matrix multiplication over adjacency matrices. Same math, more code.

**Deliverable:** `model.py`, `loss.py`, verified forward pass.

#### A3: Training on ASUS (Hours 7–11)

Build `train.py`:
- Load all training/validation graphs
- Training loop: iterate over graphs, compute loss, backprop (Adam, lr=1e-3)
- Log MSE, monotonicity penalty, BCF penalty every 10 epochs
- Save best checkpoint by validation loss
- Plot training curves → save as PNG

ASUS execution:
- SSH in, set up environment (conda + torch + pyg + CUDA)
- Transfer data, run: `python train.py --epochs 400 --hidden 64 --device cuda`
- Save checkpoint + curves
- Record ASUS hardware specs (GPU model, VRAM, training time)

**Deliverable:** `train.py`, `model.pt` checkpoint, `training_curves.png`, ASUS specs notes.

#### A4: Inference + Interpretability (Hours 11–14)

Build `inference.py`:
1. Load model + Cape Fear demo graph
2. Forward pass → tissue concentration predictions for all species-at-location nodes
3. **Interpretability extraction:**
   - GATConv attention weights from trophic edges → which prey contribute most
   - Gradient-based source attribution: `torch.autograd.grad(prediction, facility_features)` → normalize → facility contribution percentages
   - Group by factor type (source, ecological, environmental, biological, hydrologic) → contribution_pct per factor
4. **Advisory computation:**
   - Compare tissue conc to EPA reference doses (PFOS: 20 ng/g, GenX: 0.3 ng/g)
   - Safe servings/month = `ref_dose × 70kg / (tissue_conc × 0.227kg)`
   - Safety status: safe (<5), limited (5–20), unsafe (>20)
   - Recreational dose: 1 serving/month. Subsistence dose: 8 servings/month.
5. Output full JSON matching schema → `demo_data.json`

**Deliverable:** `inference.py`, `demo_data.json` ready for Track B.

---

### TRACK B: Map Visualization + Heatmap + Tooltips

**Owner:** Strongest React/frontend person
**Tools:** React (Vite), Mapbox GL JS or react-map-gl, D3.js, Tailwind CSS
**Independent until:** Hour 14 (receives `demo_data.json` from Track A)
**Depends on:** Track D delivers `mock_data.json` + `river.geojson` by hour 4

#### B1: Project Setup + Dark Map (Hours 0–2)

- `npm create vite@latest trophictrace-viz -- --template react`
- `npm install mapbox-gl react-map-gl d3 tailwindcss @tailwindcss/vite`
- Get free Mapbox API token
- Render dark-themed map centered on Cape Fear River (lat 35.05, lng -78.88, zoom 9)
- Add title bar component: "🐟 TrophicTrace" + watershed name + EJ toggle
- Set up data loading: read from local `mock_data.json` (provided by Track D)
- Basic project structure: `App.jsx`, `Map.jsx`, `Tooltip.jsx`, `DetailPanel.jsx`

**Deliverable:** React app rendering dark Mapbox map with title bar.

#### B2: River Network + Heatmap Layer (Hours 2–6)

- Load `river.geojson` (from Track D) as a Mapbox source
- Render river segments as styled lines:
  - Color by `water_pfas_ppt` (green→amber→red gradient via Mapbox `interpolate` expression)
  - Width by contamination level (2–6px)
  - Glow effect: duplicate layer at 3× width, 30% opacity, with `line-blur: 4`
- Add heatmap layer:
  - Point source at each segment centroid
  - Weight by `max_tissue_ng_g`
  - Color ramp: transparent → green → amber → red
  - Radius: 30–120px by weight
- Add facility markers:
  - Custom marker component at each facility lat/lng
  - White/yellow icon with CSS pulse animation
  - Hover → small tooltip with facility name + discharge level

**Deliverable:** Map with river network, heatmap, and facility markers rendering.

#### B3: Hover Tooltips (Hours 6–10)

- Implement Mapbox `mousemove` listener on river/heatmap layers
- On hover, find nearest segment (by geographic proximity to cursor)
- Render `<Tooltip>` component:
  - Position: offset from cursor, flip if near viewport edge
  - Dark glassmorphism card (dark bg, backdrop-blur, subtle border)
  - Content:
    - Segment name + water PFAS concentration
    - Horizontal divider
    - Species list (sorted worst-first):
      - Colored dot (8px CSS circle: red/amber/green)
      - Species name + tissue concentration
      - Advisory text (1 line)
      - "More Details →" link (blue-gray, right-aligned)
    - EJ alert box (conditionally rendered if near demographic zone, purple tint bg)
  - 150ms fade transition
  - Crossfade on segment change (no flicker)
- Wire up "More Details" click → sets selected species in React state (panel built by Track C)

**Deliverable:** Fully working hover tooltips with species safety info.

#### B4: Polish + Data Swap (Hours 14–17)

- Swap `mock_data.json` for `demo_data.json` from Track A
- Fix any rendering issues from real data (different value ranges, missing fields)
- Implement EJ overlay toggle (show/hide demographic zone polygons)
- Add map legend component (bottom-left)
- Smooth animations on initial load
- Test full hover/click flow across all segments
- Final CSS polish: shadows, transitions, spacing

**Deliverable:** Polished map visualization consuming real GNN data.

---

### TRACK C: Interpretability Panel + Detail Views + Chatbot

**Owner:** Second frontend person
**Tools:** React, D3.js (for bar chart + mini-Sankey), Tailwind CSS, (FAISS + LLM API for chatbot if time)
**Independent until:** Hour 14 (integrates with Track B's main app)
**Depends on:** Track D delivers `mock_data.json` by hour 4 (for sample data to build against)

This track builds the "More Details" interpretability panel as a **standalone React component** that can be developed and tested independently, then dropped into Track B's app at integration time.

#### C1: Panel Scaffold + Header Section (Hours 0–3)

- Create standalone React development environment (can be same Vite project as Track B, or separate — merge later)
- Build `<DetailPanel>` component that:
  - Slides in from the right (420px wide, full height, CSS transition 300ms ease-out)
  - Has dark background matching the tooltip style
  - Accepts props: `species` object (from JSON schema), `onClose` callback
  - Has a ✕ close button top-right
- Build the **header/verdict section:**
  - Species name (22px bold white) + location (14px gray)
  - Big tissue concentration number (48px bold, colored by safety status)
  - EPA reference dose comparison: "EPA limit: 20 ng/g"
  - Multiplier badge: "2.4× OVER" in red pill, or "0.3× UNDER" in green pill
  - Recommended servings: "⚠ Max 1 serving/month"
- Test with hardcoded species data from `mock_data.json`

**Deliverable:** `DetailPanel` component rendering the header section.

#### C2: Contributing Factors Bar Chart (Hours 3–6)

- Build the **"Why Is This Fish Unsafe?"** section within `DetailPanel`:
  - Section header: "WHY IS THIS FISH UNSAFE?" in 12px uppercase tracking-wide gray
  - 5 horizontal bars, sorted by `contribution_pct`
  - Each bar:
    - Left label: factor name (13px)
    - Bar: width proportional to percentage, color by factor type:
      - source: #f97316 (orange)
      - ecological: #3b82f6 (blue)
      - environmental: #14b8a6 (teal)
      - biological: #a855f7 (purple)
      - hydrologic: #6b7280 (gray)
    - Right label: percentage (13px)
    - Height: 28px per bar, 6px gap between bars
    - Rounded ends (border-radius: 4px)
  - Implementation: pure CSS with flexbox width percentages is simplest and most reliable. D3 is fine too but overkill for 5 bars.

**Deliverable:** Bar chart section rendering in the panel.

#### C3: Pathway Mini-Sankey (Hours 6–9)

- Build the **"Contamination Pathway"** section:
  - Vertical flow diagram, 4 nodes connected by lines
  - Each node is a rounded rectangle (120px × 60px) containing:
    - Label (e.g., "Chemours", "River Water", "Fish Tissue", "Human Dose")
    - Value (e.g., "450 ppt", "120.5 ppt", "48.3 ng/g")
  - Connecting lines between nodes:
    - Width increases at each step (2px → 4px → 6px) to show concentration/bioaccumulation
    - Arrow at the bottom of each line
    - Annotation next to each line: "÷3.7 dilution", "×400 BCF"
  - Node color: gradient from green (discharge, relatively "low") → amber → red (fish tissue, high)
  - Implementation: custom SVG is cleanest. Absolutely do NOT use a full D3-sankey library for 4 nodes — it's 10x the complexity for no benefit. Just manually position 4 `<rect>` elements with `<path>` connections.

**Deliverable:** Mini-Sankey SVG rendering in the panel.

#### C4: Dose Comparison Section (Hours 9–12)

- Build the **"Who Is At Risk?"** section:
  - Two horizontal bars:
    1. "Recreational angler (1 serving/mo)" → bar extends to show dose as fraction of EPA limit (e.g., 0.8×) — colored green
    2. "Subsistence fisher (8 servings/mo)" → bar extends past the EPA limit line (e.g., 2.6×) — colored red
  - A vertical dashed line at the 1.0× position, labeled "EPA limit"
  - The subsistence bar overshooting this line IS the punchline. It must be visually obvious.
  - Below the bars: text callout with community-specific info: "Subsistence fishers in Fayetteville SE (median income $31,200) face 3.2× the exposure of recreational anglers."
  - Implementation: pure CSS or simple SVG. Two `<div>` bars with width set by JavaScript, a positioned vertical dashed line.

**Deliverable:** Dose comparison section rendering in the panel.

#### C5: Integration with Track B (Hours 12–14)

- Merge `DetailPanel` component into Track B's React app
- Wire up state management:
  - Track B's tooltip "More Details" click → sets `selectedSpecies` state
  - `selectedSpecies` !== null → render `DetailPanel` with that species data + render dark scrim over map
  - Close button → clear `selectedSpecies`
- Test the full flow: hover → tooltip → click More Details → panel slides in → close → back to map
- Fix any styling conflicts between Track B's and Track C's CSS
- Add scrim overlay on the map when panel is open (dark transparent div)

**Deliverable:** Interpretability panel fully integrated into the main app.

#### C6: RAG Chatbot — STRETCH (Hours 15–18, only if core is done)

If the GNN + visualization pipeline is working:

- **Backend:**
  - Download 5–10 state fish advisories (NC, MN, MI — PDF, easily available online)
  - Convert to text, chunk into ~500-token passages
  - Embed with any available model → store in FAISS (pip install faiss-cpu)
  - Build query endpoint: question → retrieve top 5 chunks → prompt LLM with system: "You are a fish safety advisor. Use ONLY the provided context. Cite sources. If unsure, say so."
  - Also include `demo_data.json` converted to natural-language sentences as part of the corpus (so the chatbot can answer about TrophicTrace's predictions)

- **Frontend:**
  - Floating chat bubble in bottom-right corner of the visualization
  - Click to expand into chat panel (300px wide, 400px tall)
  - Simple message list + input field
  - User types → send to backend → stream response

- **Fallback (if no time for RAG):**
  - Pre-generate answers for 4–5 showcase questions
  - Display as a static "Example Queries" panel

**Deliverable:** Working chatbot OR static Q&A showcase.

---

### TRACK D: Data Pipeline, GeoJSON, Mock Data, & Integration Server

**Owner:** Full-stack / data person
**Tools:** Python, requests, JSON, GeoJSON tools (geojson.io), Flask or http.server
**Independent until:** Delivers to other tracks on a rolling basis (see schedule below)
**Depends on:** Nothing — this track has no upstream dependencies.

This person is the foundation — they produce the data artifacts that every other track needs, then become the integrator who stitches everything together.

#### D1: Cape Fear River GeoJSON (Hours 0–3) — DELIVERS TO TRACK B BY HOUR 3

This is the FIRST priority because Track B needs it to start rendering the map.

- Go to [geojson.io](http://geojson.io) or use USGS NHDPlus data
- Trace 20–30 segments of the Cape Fear River and major tributaries:
  - Main stem from ~Lillington down to ~Elizabethtown (where contamination is worst)
  - Key tributaries: Deep River, Haw River, South River
  - Each segment = a GeoJSON `LineString` feature with 10–20 coordinate pairs
- Add properties to each feature: `segment_id`, `name` (human-readable), centroid `lat`/`lng`
- Also create simplified demographic zone boundaries: 3–4 polygons (simple rectangles or 4–6 vertex shapes) for census tracts near the river, with properties: `area_name`, `center_lat`, `center_lng`

**Deliverable:** `river.geojson`, `demographics.geojson` → hand off to Track B.

#### D2: Cape Fear Research Data (Hours 0–4) — DELIVERS TO TRACK A BY HOUR 5

Compile the Cape Fear-specific data that Track A needs to build the realistic demo graph:

**Facilities** → `facilities.json`:
| Facility | Lat | Lng | Discharge (ppt) | Primary Congener | Source |
|----------|-----|-----|-----------------|-----------------|--------|
| Chemours Fayetteville Works | 34.9884 | -78.8375 | ~450 GenX | GenX (65%), PFOA (15%), PFOS (12%) | EPA ECHO + news |
| Fayetteville WWTP | ~35.05 | ~-78.88 | ~50 mixed | PFOS (40%), PFOA (30%) | Estimate from similar facilities |
| Fort Liberty (AFFF) | ~35.14 | ~-79.00 | ~100 PFOS | PFOS (70%), PFHxS (20%) | DoD PFAS reports |

**Species** → `species.json`:
| Species | Trophic Level | Lipid % | Body Mass (g) | Diet (eats what) | Source |
|---------|--------------|---------|--------------|-------------------|--------|
| Largemouth Bass | 4.2 | 5.8 | 1500 | Bluegill, Shad | FishBase |
| Channel Catfish | 3.8 | 4.2 | 2000 | Invertebrates, small fish | FishBase |
| Bluegill | 3.1 | 3.5 | 200 | Invertebrates | FishBase |
| Striped Bass | 4.5 | 6.1 | 5000 | Shad, small fish | FishBase |
| Flathead Catfish | 4.0 | 4.8 | 3000 | Other catfish, crayfish | FishBase |
| White Perch | 3.5 | 3.8 | 400 | Invertebrates, small fish | FishBase |
| American Shad | 2.8 | 5.0 | 1500 | Plankton | FishBase |
| Blue Crab | 2.5 | 1.5 | 300 | Detritus, small organisms | FishBase |

(Look up real values from fishbase.org — the above are approximate.)

**River segments** → `segments.json`:
- For each of the 20–30 GeoJSON segments: segment_id, approximate flow rate (use USGS gauge data or estimate: 20–100 m³/s on main stem, 5–20 on tributaries), pH (~7.0), temperature (~22°C summer), DOC (~5 mg/L)

**Demographics** → `demographics.json`:
- 3 zones near the river: Fayetteville SE (low-income, predominantly Black, high subsistence fishing), Fayetteville central (moderate income, mixed), rural Bladen County (low-income, mixed, high subsistence fishing)
- Use Census ACS for median income, racial composition. Subsistence fishing rates: estimate 15–25% for low-income tracts near river (based on EPA subsistence consumption study ranges)

**Deliverable:** `facilities.json`, `species.json`, `segments.json`, `demographics.json` → hand off to Track A.

#### D3: Mock Data for Track B (Hours 3–5) — DELIVERS TO TRACK B BY HOUR 5

Build `generate_mock_data.py` that creates a realistic `mock_data.json` following the JSON schema exactly:

- Use the Cape Fear GeoJSON segment coordinates for lat/lng
- Populate 20–25 segments with 4–8 species each
- Tissue concentrations: higher near Chemours (30–80 ng/g for top predators), lower upstream (1–10 ng/g)
- Contributing factors: Chemours dominant (50–70%) for segments near the facility, dilution dominant for far-downstream segments
- Pathway data with realistic numbers (discharge → dilution → BCF → tissue → dose)
- 3 facility entries, 3 demographic zones
- Safety statuses computed correctly from tissue concentrations

This mock data lets Track B develop the entire visualization without waiting for the GNN.

**Deliverable:** `mock_data.json` → hand off to Track B.

#### D4: Integration Server + End-to-End Testing (Hours 10–14)

- Build a simple serving setup:
  - Option A (simplest): Track B builds the React app → `npm run build` → serve the `dist/` folder + `demo_data.json` from a Python `http.server`. One command: `python -m http.server 8000 --directory dist/`
  - Option B (if chatbot exists): Flask app that serves the React build, the data JSON, and the chatbot API endpoint
- Write an integration validation script: load `demo_data.json`, check all required fields exist, check value ranges are reasonable, check all segment_ids in river_geojson have matching entries in segments array
- When Track A delivers `demo_data.json` (hour 14), validate it and hand to Track B

**Deliverable:** Serving setup, validation script.

#### D5: Final Integration + Demo Coordination (Hours 14–18)

- Receive `demo_data.json` from Track A
- Run validation script
- Hand validated JSON to Track B
- Help Track B/C debug any integration issues
- Test full pipeline: map loads → hover works → tooltips show real data → More Details opens with real interpretability data → EJ overlay works
- Build 5–7 slides from screenshots (minimal: hook stat, architecture diagram, live demo, impact numbers, ASUS)
- Architecture diagram: draw in Excalidraw or draw.io in 30 minutes (pipeline from data sources → graph → GNN on ASUS → heatmap)
- Record backup demo screencast in case of tech failure
- Coordinate demo rehearsal

**Deliverable:** Working end-to-end system, slides, backup video.

---

## Delivery Schedule

| Hour | Track A (GNN) | Track B (Map/Heatmap) | Track C (Detail Panel) | Track D (Data/Integration) |
|------|--------------|----------------------|----------------------|---------------------------|
| 0–3 | A1: Synthetic data gen | B1: Project setup + dark map | C1: Panel scaffold + header | **D1: GeoJSON → delivers to B** |
| 3–5 | A1: Finish data gen | B2: River + heatmap (uses D1 GeoJSON) | C2: Contributing factors chart | **D2: Cape Fear data → delivers to A** |
| 5–7 | A2: GNN architecture | B2: Finish river + heatmap | C2–C3: Finish chart, start Sankey | **D3: Mock data → delivers to B** |
| 7–10 | A3: Training on ASUS | B3: Hover tooltips | C3: Pathway mini-Sankey | D3: Finish mock data + start integration prep |
| 10–12 | A3: Finish training | B3: Finish tooltips | C4: Dose comparison section | D4: Integration server + validation |
| 12–14 | **A4: Inference → delivers `demo_data.json`** | B3: Polish tooltips | **C5: Integrate panel into Track B app** | D4: Validate + coordinate |
| 14–16 | A4: Fix any issues | **B4: Swap to real data, fix issues** | C5: Fix integration issues | **D5: End-to-end testing** |
| 16–18 | Help debug | B4: Final visual polish | C6: Chatbot (STRETCH) or polish | D5: Slides, backup video, rehearsal |
| 18–19 | All together: final demo rehearsal, last-minute fixes |

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| GNN doesn't converge | Synthetic data is generated FROM the physical model — the GNN recovers a known function. Fallback: use forward model directly for predictions, present GNN as "in training." |
| ASUS setup takes too long | Local GPU fallback. Worst case: CPU with fewer epochs + smaller graphs. Demo only needs plausible predictions. |
| PyG `HeteroConv` is buggy | Manual implementation: `torch.nn.Linear` + sparse matmul over adjacency matrices. Same math, more boilerplate. |
| Mapbox token issues | Free tier is fine. Fallback: Leaflet with OpenStreetMap tiles (no token needed, less pretty). |
| Data format mismatch at integration | JSON schema defined above is the contract. Track D validates before handing off. Mock data built from same schema ensures compatibility. |
| Interpretability panel doesn't integrate cleanly | Track C builds as a standalone React component with clean props interface. Worst case: open in a modal instead of slide-in panel. |
| Cape Fear data is hard to find | Only approximate numbers needed. Chemours contamination is extensively documented in news. Anything missing → realistic synthetic values. |
| Chatbot runs out of time | Explicit stretch goal. Demo works without it. |
| Demo crashes | Backup video recorded by hour 17. |

---

## Quick Reference

| Track | Owner | Core Output | Delivers To | Receives From |
|-------|-------|-------------|-------------|---------------|
| A | ML Engineer | Trained GNN + `demo_data.json` | B (JSON, hr 14) | D (Cape Fear data, hr 5) |
| B | Frontend #1 | Map + heatmap + tooltips | Demo | D (GeoJSON hr 3, mock data hr 5), A (real data hr 14), C (panel hr 14) |
| C | Frontend #2 | Interpretability panel + (chatbot) | B (component, hr 14) | D (mock data hr 5) |
| D | Data/Integration | GeoJSON, mock data, Cape Fear dataset, integration | A (data hr 5), B (GeoJSON hr 3, mock hr 5), all (integration hr 14+) |
