# Concepts primer: the project, explained from scratch

This is for someone new to drug combinations, active learning, or LLM agents. It explains the ideas
behind [`../target_design.md`](../target_design.md) intuitively, with pictures. Technical claims here
restate the design doc. Where I add something of my own, I say so.

---

## 1. The biology in one picture: why combine drugs?

```
   Drug A alone          Drug B alone          A + B together
   kills 30% of cells    kills 30% of cells    kills ??? of cells

   If they don't interact ("independent"), we EXPECT about 51% killed
   (each drug spares 70% of cells, and 0.7 × 0.7 = 0.49 survive).

   Observed 80% killed  → SYNERGY      (better than expected)   ← what we want to find
   Observed 51% killed  → additive     (as expected)
   Observed 35% killed  → ANTAGONISM   (worse than expected)
```

- The "independent" calculation above is the **Bliss** model. It is the design's primary synergy
  score, `y_syn`.
- ZIP, Loewe and HSA are other definitions of "expected," and they give somewhat different numbers.
  The design keeps them as secondary checks.
- **Growth inhibition** (`y_inh`, measured by CSS) asks a different question: how well does the pair
  kill cells, *regardless* of whether that's more than expected?
  - A pair can be strongly synergistic but weak overall (two weak drugs that help each other a lot).
  - A pair can kill well with no synergy (one very strong drug).
  - That's why the design tracks the two outcomes separately.

The same pair behaves differently in different **cell lines** (cancer cells grown in a lab, each from
a different tumor). One **experiment** is therefore a **triplet**:

```
   (drug A, drug B, cell line)   e.g. (trametinib, alpelisib, a KRAS-mutant lung line)
```

## 2. The problem: far too many experiments

The primary benchmark (TDC DrugComb, from the NCI-60 screen) has about **129 drugs × 59 cell lines**.
That's ~297,000 measured triplets according to TDC's metadata.

A real lab can afford a few hundred. The design's budget is **320 = 20 rounds × 16**, about 0.1% of
the pool. The question becomes: **which 320 should we run to find as many strong combinations as
possible?**

## 3. The simulated lab (the "oracle"): how we test this without a wet lab

The trick: every result in the dataset has *already* been measured. We hide them all and pretend we
don't know them.

```
     ┌──────────────────────── Oracle (simulated lab) ────────────────────────┐
     │  hidden table: 297k triplets → (y_syn, y_inh)      🔒 only run() reads   │
     └───────────────▲──────────────────────────────────────────┬─────────────┘
          "run these │16 triplets"                       results │ for just those 16
                     │                                           ▼   (budget −16)
     ┌───────────────┴─────────────── Selector ───────────────────────────────┐
     │  random  |  uncertainty  |  Bayesian optimization  |  our LLM agent    │
     └────────────────────────────────────────────────────────────────────────┘
```

- Every selector plays the same game:
  - same candidate pool;
  - same 32 free "seed" results at the start;
  - same budget;
  - same batch size.
- The only difference between selectors is **how they choose**. This makes the comparison fair.
- **Held-out cell lines.** About 20% of lines are hidden from every selector for the whole game.
  They're used only at the end, to check whether the agent's mechanism claims predict anything (§6).

## 4. Active learning and Bayesian optimization (the baselines)

Every selector uses the same **response model**: a statistical model that guesses the result of any
untested triplet, *and says how unsure it is*.

```
predicted synergy
   ▲                       ╭─╮  ← big uncertainty: "we've never tried anything like this"
   │        ●  ╭──╮       │ │
   │   ●  ╭─┴──┴─╮│  ●    │ │      ● = experiments already run (known)
   │ ╭────┤ mean ├┴──────────┴──╮   ─ = model's best guess (mean)
   │ │    ╰──────╯   small band  │   band = ± uncertainty (std)
   └──────────────────────────────────────►  candidate triplets
```

Different selectors use the model differently:

| Selector | Picks | Intuition |
|:--|:--|:--|
| `random` | any untested triplet | the floor; any real method should beat it |
| `greedy` | highest **mean** | "go where it looks best": can get stuck |
| `uncertainty` | highest **std** | "go where we know least": learns, but doesn't hunt for hits |
| `bo_ucb` | highest **mean + 2 × std** | balances both: the standard Bayesian-optimization move |
| `bo_qei` | best expected improvement for a whole batch | a stronger, batch-aware BO |

This balance is the **explore/exploit trade-off**. Every round: run a batch, refit the model, choose
again.

## 5. One agent round

The agent adds an LLM that reasons about *mechanisms*. An example of that kind of reasoning: "MEK
inhibitors plus PI3K inhibitors should synergize in KRAS-mutant cells, because blocking one pathway
makes cells lean on the other."

Hard rule from the design: **the LLM never outputs numbers.** All predicted values come from the
statistical model. The LLM only decides *what to test* and *why*.

Below is my **reconstruction** of the round, from the 9 steps written in design §8. The original
figure was lost (`target_design.md:223`).

```
 ┌──── no LLM (deterministic code) ───────────────────────────────────────────────┐
 │ 1 Refit model on all revealed results → score every candidate                    │
 │ 2 Flag surprises: |observed − predicted| / std > 2  → "surprise" observations    │
 │ 3 Shortlist: top-200 by UCB + top-50 by std + 50 random from untested regions   │
 └───────────────────────────────┬────────────────────────────────────────────────┘
                                 ▼
 ┌──── LLM (reasoning, via tools) ────────────────────────────────────────────────┐
 │ 4 Explore      propose / revive hypotheses ("pair P works in context C via M")   │
 │ 5 Select       pick ≤ 5 hypotheses to test this round                            │
 │ 6 Candidate    gather supporting / refuting evidence (literature, drug DB,       │
 │   analysis     cell genomics, model queries)                                     │
 │ 7 Comparative  design experiments that tell rival hypotheses apart               │
 │   analysis     (e.g. same pair in mutant vs wild-type lines)                     │
 └───────────────────────────────┬────────────────────────────────────────────────┘
                                 ▼
 ┌──── mixed ─────────────────────────────────────────────────────────────────────┐
 │ 8 Assemble batch of 16: ≥4 slots forced from top-UCB, rest chosen by LLM, each  │
 │   citing a hypothesis; a critic call checks redundancy/leakage; revise once     │
 │ 9 oracle.run(batch) → new observations → mark hypotheses retained / set aside   │
 └──────────────────────────────────────────────────────────────────────────────────┘
          if the LLM's output fails schema checks twice → fall back to bo_ucb this round
```

**SHELDON's key idea: two spaces.** SHELDON is the reasoning harness we build on. It keeps
*hypotheses* (claims) and *observations* (evidence) as two separately organized maps. Every
(hypothesis, observation) pair is labeled supports, refutes, or neutral.

```
   HYPOTHESIS SPACE                         OBSERVATION SPACE
   regions: MAPK×PI3K | DNA-damage | …      regions: revealed results | literature | genomics
     H1 "MEK+PI3K synergize in KRAS-mut"  ──supports──  O7 "trametinib+alpelisib Bliss 14 in A549"
     H2 "…in all lung lines"              ──refutes───  O9 "same pair Bliss −2 in a KRAS-wt lung line"
```

(The H/O entries and numbers above are made up to illustrate the idea. They are not data.)

SHELDON's paper reports that treating observations as first-class objects (not just raw input)
makes the agent look for evidence that separates hypotheses. Removing the observation space was the
largest ablation loss in their results table. The design's `agent_single_hyp` variant tests whether
that idea helps *here*.

## 6. The secondary question: do the mechanisms predict anything?

When the agent claims a *mechanistic* reason for a hit, it must register a **checkable rule** before
anyone looks at the held-out lines:

```
   Hypothesis: "MEK+PI3K synergize because of KRAS-driven MAPK dependence"
   Registered rule:  KRAS_mut == 1      direction: rule-true lines have HIGHER synergy

   Held-out lines (never seen during the game):
     KRAS_mut = 1 :  Bliss  12, 9, 15, 7      ┐  compare the two groups:
     KRAS_mut = 0 :  Bliss  1, -3, 4, 2, 0    ┘  AUROC, Mann-Whitney test, FDR correction
```

A rule only "counts" if it beats a **dumb baseline rule** that code mines from the training data
with no reasoning at all. Otherwise the LLM merely rediscovered an obvious correlation.

## 7. Leakage: three ways to cheat by accident

```
  1 Data access     a tool accidentally returns hidden labels   → only oracle.run() reveals; tests scan logs
  2 Papers          the agent reads the paper that published    → blocklist the screen papers and reanalyses
                    this exact screen's results
  3 LLM memory      the model memorized famous results          → can't remove it, so MEASURE it:
                                                                   famous vs obscure split, zero-shot probe,
                                                                   drug names replaced by codes
```

Route 3 is the subtle one. If the agent wins only on famous pairs, and loses its edge when drug names
are hidden, it was *remembering*, not *reasoning*.

My note: this matters a lot here, because both benchmarks are famous public screens (NCI-ALMANAC and
O'Neil/Merck). See [target-design-gap-analysis.md §2.4](target-design-gap-analysis.md).

## 8. How we'll know who won

```
 cumulative hits
   ▲                                  ╭──── agent?
   │                           ╭──────╯ ╭── bo_ucb
   │                    ╭──────╯ ╭──────╯
   │             ╭──────╯╭───────╯ ╭───────── uncertainty
   │      ╭──────╯───────╯ ────────╯ ··········· random (≈ 5% hit rate)
   └──────┴───────┴───────┴───────┴──────►  experiments run
          80     160     240     320
```

- **Primary endpoint:** synergy hits by 320 experiments, agent vs `bo_ucb`, paired over 10 seeds.
- Supporting metrics:
  - area under this curve;
  - experiments needed to reach 10 hits;
  - diversity (how many *different* pairs);
  - cost per hit;
  - everything split into famous, middle, and obscure pairs.

## 9. Glossary cheat-sheet

| Word | Plain meaning |
|:--|:--|
| Triplet | one experiment: (drug A, drug B, cell line) |
| Pool | all triplets any selector may choose from |
| Oracle | the simulated lab that reveals hidden results |
| Hit | a triplet in the top 5% for that outcome |
| Response model | the statistics that predict results with uncertainty |
| UCB | "upper confidence bound" = mean + β·std; an optimistic guess |
| Seed set | 32 free results everyone starts with (fixed per random seed) |
| Held-out lines | cell lines nobody may select; used only to test mechanism rules |
| Leakage | any way the answer sneaks into what the selector sees |
