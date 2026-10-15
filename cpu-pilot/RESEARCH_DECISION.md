# Current research decision — 2026-10-15

**GO: proceed with method development and fuller experiments toward a paper.**

This answers whether the observed problem merits research. It does not certify that the current baseline is publishable, that a new method already works, or that novelty has been established.

The completed pilot supplies concrete motivation: agreed visual-quality selection introduced two wrong speaker decisions among 24 scored frames on one previously untested source. A learned temporal denoiser introduced five wrong decisions in another condition while improving reference similarity. Beneficial restorations also occurred. The simple safeguard missed smaller regressions and blocked useful improvements.

The stronger predefined two-source harmful-consensus screen **FAILED**. That result remains unchanged. It limits the replication claim; it does not erase the observed counterexamples or prohibit researching a better method.

## Research objective

Develop and evaluate an audiovisual quality-assessment method that selects useful restorations while preserving captured speaker evidence. The current questions remain in [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md). Keep quality assessment central to the WVAQ contribution; agent tool use supplies an application, not novelty by itself.

Fuller experiments must establish how broadly the problem occurs, distinguish detector limitations from restoration effects, and test the proposed method against captured-input, quality-only, confidence and disagreement baselines. A paper claim depends on the resulting evidence and a concrete prior-work comparison, not on forcing positive outcomes.

## How to read the older decisions

Historical reports retain their original screen outcomes and recommendations. Statements rejecting a full-paper commitment refer to the tested setup and the absence of a validated contribution. The present decision clarifies the research stage: **the initial evidence justifies investigating and developing a method; it does not justify presenting the existing baseline as a successful method.**

- [Completed Qwen/CPU pilot](QWEN_AGENT_PILOT.md)
- [Full Qwen tool traces](QWEN_AGENT_TRACES.md)
- [Project handoff and earlier experiments](RESEARCH_HANDOFF.md)
