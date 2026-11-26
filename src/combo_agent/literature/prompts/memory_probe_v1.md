You are an expert cancer pharmacologist. You will be given drug-pair experiments from a
high-throughput in-vitro screen: two approved or investigational cancer drugs tested together on one
human cancer cell line (the NCI-60 panel), across a dose matrix.

For each experiment, estimate from your own knowledge (you have no tools and no data):

- `synergy`: how synergistic the pair is in that cell line, i.e. how far the combined effect exceeds
  what the two drugs would do independently (Bliss independence). 0 = clearly antagonistic or no
  extra effect, 50 = typical for this screen, 100 = among the most synergistic experiments in it.
- `killing`: how strongly the combination inhibits growth of that cell line overall (a combination
  sensitivity score). 0 = barely any growth inhibition, 50 = typical, 100 = among the most potent.

Most experiments in such a screen are near-additive, so use the whole scale to rank them: give high
scores only where you have a specific reason. Return one entry for every experiment id, and no
others.

Experiments:

{experiments}
