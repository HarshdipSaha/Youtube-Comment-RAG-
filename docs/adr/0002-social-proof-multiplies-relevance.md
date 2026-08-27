# ADR 0002 — Social proof multiplies relevance; it does not add to it

**Status:** accepted (revised once — see History)

## Context

A cluster is ranked on three signals: how well it matches the question
(`salience`), how many people hold it (`support`), and how many likes they drew
(`endorsement`). They have to be combined somehow.

## Decision

    score(C, q) = relevance(C, q) × social(C)

    relevance(C, q) = (1 - γ) + γ · saliencẽ(C, q)
    social(C)       = 1 + α · support̃(C) + β · endorsement̃(C)

Defaults: γ=0.6, α=0.2, β=0.2. Consensus questions use γ=0.25, α=0.4, β=0.35.

## Rationale

The obvious form is additive — `γ·salience + α·support + β·endorsement` — and it
is wrong in a way that is easy to miss and easy to measure. With α+β=0.4, a
maximally-supported cluster scores 0.4 from social terms alone, so it beats any
cluster whose rescaled salience falls below 0.67. In practice that means asking
about a minor character returns the video's most-liked joke. Popularity was
substituting for relevance rather than modulating it.

Multiplying fixes it structurally. An irrelevant cluster has `saliencẽ ≈ 0`, so
`relevance ≈ 1-γ`, and no amount of social proof lifts it past a genuinely
relevant cluster. Among clusters that *are* relevant, the more widely-held one
wins. Relevance decides *whether*; consensus decides *which*.

γ appears as a mixing fraction rather than a coefficient, so it is constrained to
[0, 1] and reads directly: γ=1 annihilates off-topic clusters, γ=0 ignores
relevance entirely — which is exactly what `overview()` wants, since there is no
question for anything to be relevant to.

## Consequences

- The weights mean what they say, and the validator enforces γ ≤ 1.
- Ranking is not a linear model, so it cannot be fitted by linear regression if
  learned weights are ever wanted. An acceptable trade for correct semantics.

## History

Salience was originally the **maximum** similarity over cluster members. That is
not size-neutral: a six-member cluster gets six draws at that maximum and a
three-member cluster gets three, so the larger group scored higher on noise
alone. Since `support` and `endorsement` already reward size deliberately,
letting salience reward it accidentally double-counted it. Salience is now
measured once per cluster — see ADR 0004.
