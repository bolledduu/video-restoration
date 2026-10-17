# Run the audiovisual evidence experiments in Antigravity

Antigravity is the execution IDE. Its coding assistant is not automatically the evaluated agent. The evaluated agent is the exact API model passed to the runner. Changing IDE assistants must not change the experimental protocol.

## Completed smoke validation

The final `interval_v1` run completed all three clips from one source with `qwen/qwen3.8-27b`, answering all 297 scored timestamps. Agent errors were 62, 0 and 45, compared with 73, 0 and 45 for captured-input detector predictions. The tested restorers introduced no correct-to-wrong transitions in this batch. Sixteen software checks passed. See `results/ava_smoke_v1/RESULTS.md` and `EXECUTION_NOTES.md` for measured results, the inaccurate Qwen comparison, and earlier interface failures. This validates the runnable AVA workflow; it does not complete the full research experiments.

## Ready environment

The implemented dataset adapter is AVA-ActiveSpeaker v1.0 validation. Four source videos and 11 eligible four-second windows are frozen locally. The current smoke run uses only the first source and its three windows. AVA's official speaking labels and face tracks are evaluator-only. AMI and EasyCom are not implemented adapters in this environment; do not pretend they are interchangeable with AVA or claim results for them.

The local CPU stages create controlled noisy face crops, restore them with FastDVDnet and spatial non-local means, measure MUSIQ and BRISQUE, and run TalkNet on each variant with unchanged audio. The agent can list evidence, inspect quality, inspect speaking scores, inspect intervals, compare with capture, and submit decisions or abstentions. It receives measured evidence, not audiovisual files. This is an instrumented agent baseline, not the proposed research method.

The selected clips are frozen before inference. Do not replace clips after observing favorable or unfavorable results. Gaussian noise is an explicit controlled condition; it is not a real transmission fault. Original AVA source images are not pristine references.

## Local prerequisites

Open `video-restoration` in Antigravity on this laptop. Keep the sibling `agent-audio-visual` directory: it contains the downloaded media, third-party source, weights, and installed CPU packages. This handoff reuses that verified local runtime; it is not a claim that a fresh machine needs no setup. `results/ava_smoke_v1/runtime_receipt.json` records runtime hashes and package versions. Do not upload source movies or credentials to Git.

Use `GROQ_API_KEY` for Groq or `GEMINI_API_KEY` for Gemini, stored outside the repository. Windows User environment values are supported. The optional backup credential is used only for authentication failure, never to bypass rate limits. No model is silently substituted. The runner checks the provider's model inventory before inference. Gemini's configured model identifier is `gemini-3.8-flash`; availability must be checked with the actual account. The Gemini adapter has not been live-tested without its credential.

From the repository root in PowerShell:

```powershell
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage check
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage measure -Sources 1 -RunName ava_smoke_v1
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage agent -Provider groq -Model qwen/qwen3.8-27b -RunName ava_smoke_v1
python cpu-pilot/scripts/report_agent_experiment.py --run-dir cpu-pilot/results/ava_smoke_v1
```

Completed measurements and completed agent episodes resume without repeating inference. Changed code, model, measurements or protocol require a new output directory/file. Incomplete agent episodes are preserved and require an explicitly separate retry, not silent overwriting. API errors and budget exhaustion are not model answers. Each episode permits at most eight model calls and 24 tool calls.

The current transport configuration limits each response to 768 tokens. Speaking inspections return compact `[start, end_exclusive, speaking]` intervals that preserve every 0.5 threshold decision. The agent can inspect up to 20 detailed scores per interval; these are truncated to six decimals without changing threshold decisions. Earlier failed requests are retained separately: long score arrays and repeated inspections exceeded the account's input-token allowance. They are infrastructure failures, not negative model predictions, and are not pooled with the final compact-tool run. Use `-Attempt new_label` to create a distinct retry artifact; do not reuse an incomplete filename. Avoid simultaneous runs writing the same output path.

The compact_v1 trial exhausted eight model turns while repeatedly inspecting evidence. The bounded_v1 trial forced submission but Qwen repeatedly supplied only 32 decisions for 99 timestamps; these were correctly rejected. The final interval_v1 interface accepts consecutive `[start, end)` ranges and expands only the model's stated decisions after checking complete coverage without gaps or overlap. Its controller forces the submission tool once all candidates have been inspected and restored candidates compared, starting at turn five; it also forces submission at the final eighth turn. It never chooses the candidate or fills missing answers. These are declared interface/controller policies, not evidence that unassisted Qwen always finishes. Preserve earlier unfinished attempts when reporting operational failure rates.

To evaluate another model on exactly the existing measurements:

```powershell
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage agent -Provider gemini -Model gemini-3.8-flash -RunName ava_smoke_v1
python cpu-pilot/scripts/report_agent_experiment.py --run-dir cpu-pilot/results/ava_smoke_v1
```

To extend to the four already frozen sources, first approve the larger API budget, then use a new run name:

```powershell
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage measure -Sources 4 -RunName ava_four_sources_v1
./cpu-pilot/scripts/run_agent_experiment.ps1 -Stage agent -Provider gemini -Model gemini-3.8-flash -RunName ava_four_sources_v1
python cpu-pilot/scripts/report_agent_experiment.py --run-dir cpu-pilot/results/ava_four_sources_v1
```

Four sources are still an integration batch, not the final study. Before a publication-scale run, freeze source-disjoint development/test splits, sample counts, restoration conditions, model versions, call budget and the primary endpoint. Do not tune a safeguard on these pilot clips and report it as held-out performance. Add additional dataset adapters only after validating each dataset's timestamp, person identity and audible-speaking annotation semantics.

## Paste this into the Antigravity assistant

Run the existing experiment code in this repository. Read cpu-pilot/AGENTS.md and cpu-pilot/ANTIGRAVITY_HANDOFF.md first. Do not redesign the method, modify the frozen selection, tune thresholds, replace unsuccessful clips, or write paper claims. Start with the check command. Reuse the existing ava_smoke_v1 measurements and completed agent results. Run only the model and sample budget I explicitly request. Keep the coding assistant separate from the evaluated API model. Use credentials only from the environment. If a check fails, preserve the failed artifact and explain the error; do not bypass the check or fabricate results. Record exact requested/served model identifiers, tool traces, token usage, hashes, exclusions and completion status. Generate RESULTS.md using the report script. Report correct-to-wrong and wrong-to-correct changes separately, and compare agent errors with captured-input errors on the same answered timestamps. Never count correlated frames as independent samples. State which results are measured and which experiments remain unrun. Do not commit credentials or media. Ask before changing the protocol or increasing paid usage beyond my specified budget.
