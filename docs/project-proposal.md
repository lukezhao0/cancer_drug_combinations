### Literature-Informed Experiment Selection for Cancer Drug Combinations

**ID:** Cancer-Combo-Agent

**Level:** Advanced

**Domain:** Biomedical

**Proposed advisor:** Dr. Bryant Lin

**Level: Advanced**

**Domain**: Biomedical

**Mentor:** Bryant Lin, MD

**Other Advisors:** Minseung Choi, Janina Troper

**Question:** Can an agent that combines literature evidence with a statistical model find effective drug combinations for cancer cells in fewer experiments than standard active learning, and can its proposed mechanisms generate correct, testable predictions about which cancer cells will respond?

**Motivation:** Cancers that become resistant to one drug are often treated with combinations, but the number of possible pairs is far too large to test. Labs pick experiments by intuition and literature reading, one batch at a time. Standard active-learning methods pick experiments from numbers alone and ignore what is known about drug mechanisms. This project tests whether an agent that reasons like a scientist \- proposing hypotheses from the literature, then updating after each result \- finds good combinations faster. It also asks whether the agent's proposed mechanisms generate accurate predictions about which cells will respond. The agent is benchmarked on a simulated lab: a dataset where every result has already been measured but is hidden until the agent "runs" that experiment.

**Project:**

Data. Use drug-combination screens from Therapeutics Data Commons and the DrugComb portal, restricted to cancer cell lines. Pool the cancer lines and fix a candidate pool of drug pairs.

Simulated lab. Start with a set of drug-cell-line experiments that have measured results. Hide those results from the agent. In each round, the agent selects a small number of experiments to “run,” sees their results, and uses them to decide which experiments to run next.

Agent. A numerical response model with uncertainty estimates proposes candidates. The LLM retrieves mechanism evidence (SHELDON), proposes and critiques hypotheses, and chooses the next batch. Statistical tools, not the LLM, estimate responses.

Explanations. For each combination the agent flags as promising, it writes an explanation and labels its level: mechanistic (a specific pathway in these cells), cross-cancer (worked in other tumor types), or statistical (similar drugs behaved this way). For mechanistic explanations, the agent must state a testable prediction about which cell lines should respond, based on the proposed mechanism.

Leakage control. Keep hidden measurements, and any paper reporting those exact screen results, out of the agent's evidence. Measure separately how the agent does on famous combinations versus obscure ones, since the LLM may already know the famous ones.

**Evaluation:**

Discovery efficiency: number of strong combinations found for a fixed experiment budget, compared against random selection, uncertainty sampling, and Bayesian optimization.

Two outcomes: report both synergy (the pair does better than expected from each drug alone) and overall growth inhibition (the pair kills cells well), since they answer different questions.

Explanations: test whether the predictions implied by the agent's proposed mechanisms hold on held-out cell lines.

**Prerequisites**: Python, machine learning (Bayesian optimization or active learning is a strong plus), and experience building LLM agents. Basic cancer biology is helpful.

**References**: Therapeutics Data Commons (Huang et al., 2021); DrugComb (Zagidullin et al., 2019; Zheng et al., 2021); NCI-ALMANAC (Holbeck et al., 2017); O'Neil et al., 2016; SHELDON; DataSTORM (Liu et al., COLM 2026).

**Quarter Goal**: A working agent on a fixed cancer-cell-line candidate pool, efficiency curves against the three baselines for both synergy and growth inhibition, and an analysis of whether proposed mechanisms make accurate predictions on held-out lines.

**Status**: New

**Project page**:
