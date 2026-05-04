# Simulated Review Record

## Round 1 Decision: Major Revision

Reviewer A (novelty and method) found that the first draft needed a clearer
separation from PEXP. The revision added a dedicated source-paper reading
section and positioned OTBE as a simulation-grounded method rather than an
image-specific PEXP variant.

Reviewer B (experiments and reproducibility) found that the first draft did
not define every metric in the manuscript. The revision added equations for
top-k Jaccard, repeated-run stability, attribution RMSE, and spurious mass,
and tied the tables to generated experiment artifacts.

Reviewer C (writing and positioning) found that the first draft overclaimed
the results. The revision now states that OTBE is strongest on repeated-run
stability while remaining competitive, not universally superior, on attribution
accuracy.

## Round 2 Decision: Minor Revision

Reviewer A requested a clearer statement that the covariance correction is not
causal discovery. The discussion now states this boundary explicitly.

Reviewer B requested evidence that the experiments are simulation-only and
reproducible. The reproducibility checklist now points to the experiment script,
manifest, raw metrics, summary metrics, statistical tests, and figures.

Reviewer C requested removal of all non-English text. The author note was
changed to English and the LaTeX source no longer loads Chinese text packages.

## Final Simulated Editorial Decision: Accept

The final manuscript is internally consistent: the method, experiment code,
figures, tables, and claims all support the bounded contribution. Remaining
external-validity limits are disclosed rather than hidden.

