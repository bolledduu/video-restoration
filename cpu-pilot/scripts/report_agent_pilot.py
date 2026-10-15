"""Generate a complete small-pilot report without turning discovery into confirmation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def main():
    r=json.loads((ROOT/'results/agent_pilot_measurements.json').read_text())
    ami=json.loads((ROOT/'results/ami_agent_pilot.json').read_text())
    qp=ROOT/'results/qwen_agent_pilot.json';q=json.loads(qp.read_text()) if qp.exists() else {'cases':[],'model':'not run'}
    completed=[c for c in q['cases'] if c['status']=='completed']
    lines=['# Qwen audiovisual evidence pilot — 2026-10-15','',
    '**Primary screening decision: FAIL. Do not commit to a full paper on the strength of this pilot.** The fixed two-source replication threshold was not met. There are real measured failures worth studying, but they coexist with improvements and clean-input detector failures. This is not a disproof of the research questions.','',
    '## What actually ran','',
    '- Two previously untested UniTalk validation sources, four one-second tracks balanced speaking/silent, two fixed Gaussian-noise levels: eight paired cases. Four processing candidates plus clean references produced 40 local model evaluations, with 120 sampled-frame evaluations per IQA metric.',
    '- Official learned five-frame FastDVDnet clipped-noise weights on CPU, at the true noise level and fixed sigma 55; conventional spatial NLM control; captured input always available. TalkNet uses unchanged original audio. BRISQUE and MUSIQ assess the same three frames per candidate.',
    '- Separate AMI ES2002a meeting: one speaking and one listener segment, three variants each, six local evaluations. Published transcript segments are proxy activity labels; they are not independently annotated dense ASD labels. This meeting is reported separately, not pooled into the primary screen.',
    f'- Qwen model verified through Groq account API: `{q["model"]}`. Completed {len(completed)}/8 tool-agent cases. Tools expose cached real measurements; Qwen does not directly see/hear media. It chooses evidence and answers majority speaking status, not a full quotation/transcription task.',
    '- No labels, source names, reference images, SSIM, error rates, or oracle choices enter Qwen tool responses. They are used only by the offline evaluator. No credential is saved in scripts or result files.',
    '- Noise is added after resizing face crops to 128x128. Every clip is one second; results are not benchmark mAP or a test of longer-context TalkNet performance. Sources are new to this project\'s experiments, not certified absent from every pretrained model\'s training set.','',
    '## Primary screening rule and outcome','',
    'Before inference: require harmful BRISQUE/MUSIQ agreement on at least two independent sources, plus useful restoration on at least one source. Harm means at least one additional wrong frame among 24 scored frames. All eligible samples are reported; none were replaced based on outcomes.','',
    f'- Agreed harmful selection: **{len(r["summary"]["agreed_harmed_sources"])} / 2 sources**.',
    f'- Restoration improves speaker error: **{len(r["summary"]["restoration_helped_sources"])} / 2 sources**.',
    '- Result: **FAIL**, because harmful consensus did not replicate across both sources.','',
    '## Selection performance','',
    'These are means over eight paired cases from only two independent sources. Frames and noise variants are not independent participants. Confidence is an uncalibrated model score. Oracle selection uses ground truth and is not deployable.','',
    '| Selector | Frame error | Mean MUSIQ (higher better) | Mean BRISQUE (lower better) | Mean reference SSIM |','|---|---:|---:|---:|---:|']
    for name,value in r['summary']['mean_error'].items():
        scores=[c['candidates'][c['winners'][name]] for c in r['cases']]
        lines.append(f'| {name} | {100*value:.2f}% | {np.mean([v["musiq"] for v in scores]):.2f} | {np.mean([v["brisque"] for v in scores]):.2f} | {np.mean([v["ssim"] for v in scores]):.3f} |')
    if len(completed)==8:
        lookup={c['id']:c for c in r['cases']};scores=[lookup[c['id']]['candidates'][c['submission']['candidate']] for c in completed]
        lines.append(f'| Qwen | {100*np.mean([c["evaluation"]["selected_frame_error"] for c in completed]):.2f}% | {np.mean([v["musiq"] for v in scores]):.2f} | {np.mean([v["brisque"] for v in scores]):.2f} | {np.mean([v["ssim"] for v in scores]):.3f} |')
    lines+=['','## Every primary case','','| Source / true state / noise | Captured error | BRISQUE choice / error | MUSIQ choice / error | Guard error |','|---|---:|---|---|---:|']
    for c in r['cases']:
        v=c['candidates'];w=c['winners'];lines.append(f'| {c["source"]} / {c["label"]} / {c["sigma"]} | {v["captured"]["error"]*100:.2f}% | {w["brisque"]} / {v[w["brisque"]]["error"]*100:.2f}% | {w["musiq"]} / {v[w["musiq"]]["error"]*100:.2f}% | {v[w["guard"]]["error"]*100:.2f}% |')
    lines+=['','## Where failures occur, and what explains them','','1. **Agreed harmful conventional restoration:** EEey6Pt0rqI, silent, sigma 15. Both assessors select spatial NLM. SSIM rises 0.387 → 0.889, BRISQUE improves 43.66 → 37.23, and MUSIQ improves 13.84 → 17.82, but errors rise 0/24 → 2/24. The simple disagreement guard allows this because 2/24 is below its fixed 0.1 threshold. This is a directly measured counterexample to treating metric agreement as a guarantee of task preservation.',
    '2. **Learned temporal restoration failure:** the same silent source at sigma 40. FastDVDnet raises SSIM 0.117 → 0.927 and MUSIQ 13.74 → 15.96, but errors rise 0/24 → 5/24. MUSIQ selects it; BRISQUE selects spatial NLM. The guard retains captured video and avoids these five errors. This establishes a visual-processing effect on TalkNet under fixed audio, not audio-caused hallucination or proof that the restored person visibly speaks.',
    '3. **Clean-reference detector failure:** kgVz_AXYvn0 speaking track is misclassified in all 24 clean frames. At sigma 40, noise happens to make the captured prediction correct; every denoiser returns it to all-wrong predictions. Therefore that dramatic 0% → 100% change cannot by itself establish destruction of genuine speaking evidence. It may reflect detector sensitivity and removal of noise that accidentally helped the detector.',
    '4. **Beneficial restoration is present:** on kgVz_AXYvn0 silent sigma 15, the best candidate reduces errors 8/24 → 1/24. At sigma 40, it reduces 3/24 → 0/24. The fixed guard blocks these improvements because it treats any substantial prediction change as risky. This is why retaining the input is not a complete solution.',
    'The intervention identifies which processing changes predictions. It does not identify the exact mouth-motion mechanism, verify human perceptual preference, or establish calibrated confidence. Clean controls, unchanged audio, and counterexamples constrain the explanation. The two agreed-selection false detections have only 0.512–0.513 speaker scores; the five learned-restoration false detections have 0.526–0.600 scores. These are not evidence of highly confident wrong attribution. Gaussian corruption is a controlled synthetic intervention, not an evaluation of all natural camera degradations.','',
    '## Separate AMI meeting results','','| Transcript activity proxy | Captured error | FastDVDnet error | Clean-reference error | Captured → restored MUSIQ |','|---|---:|---:|---:|---:|']
    for c in ami['cases']:
        v=c['candidates'];lines.append(f'| {c["label"]} | {v["captured"]["error"]*100:.2f}% | {v["fastdvdnet"]["error"]*100:.2f}% | {v["reference"]["error"]*100:.2f}% | {v["captured"]["musiq"]:.2f} → {v["fastdvdnet"]["musiq"]:.2f} |')
    lines+=['','No attribution change was observed in this meeting. Both quality metrics and SSIM improved. TalkNet missed the speaking segment even in the clean reference. The listener has a hand near the mouth; that natural occlusion was retained. The fixed camera/person mapping comes from AMI metadata. A face-crop amendment was made after Haar failed, using original frames only and before any AMI model inference.','',
    '## Qwen agent results','','The agent has a bounded tool menu: list candidates, inspect quality/speaker measurements, compare with capture, and submit a selection with a speaking decision and explanation. There is no browsing, arbitrary code execution, or access to evaluation labels. This tests evidence selection over tool outputs, not autonomous restoration-code generation.','',
    '| Case | Selected evidence | Majority answer correct | Selected frame error | Reported confidence |','|---|---|---|---:|---:|']
    for c in completed:
        s=c['submission'];e=c['evaluation'];lines.append(f'| {c["id"]} | {s["candidate"]} | {e["majority_answer_correct"]} | {e["selected_frame_error"]*100:.2f}% | {s["confidence"]:.2f} |')
    if completed:
        lines+=['',f'Completed majority-state answers: {sum(c["evaluation"]["majority_answer_correct"] for c in completed)}/{len(completed)} correct. This binary clip-level endpoint is coarser than per-frame attribution: a clip answer can be correct despite several wrong frames.']
        usage=[u for c in completed for u in c['usage']]
        lines += [f'Recorded successful-call usage: {sum(u.get("prompt_tokens",0) for u in usage):,} input tokens and {sum(u.get("completion_tokens",0) for u in usage):,} output tokens. Raw messages, tool calls, measurements, decisions, and usage are retained in `results/qwen_agent_pilot.json`. Failed retries are not included in these usage totals.']
    lines+=['','### Explanation audit','',
    'Qwen explanations require checking against tool outputs. In the first completed case, Qwen described only 3/24 captured scores above 0.5; the recorded scores actually contain 8/24. Its final majority-state answer was still correct. A plausible explanation is not itself evidence, and no numerical claim in an agent explanation should replace the offline evaluator.','',
    '## Implications for the existing research questions','',
    '- **RQ1: concrete but limited supporting evidence.** Harmful assessor agreement occurs on one new source, and a learned temporal restorer also changes correct speaker predictions to wrong ones. The stronger predefined two-source replication screen fails. Agreement is not shown to predict harm generally.',
    '- **RQ2: the tested simple safeguard is insufficient.** Disagreement catches the larger learned-restoration failure but misses a smaller agreed failure and blocks beneficial changes. This motivates investigating evidence preservation; it does not validate a novel assessor.',
    '- **RQ3: no demonstrated winning selector.** MUSIQ improves pooled frame error over retaining captured input, while the guard does worse. A selector must preserve these improvements while reducing regressions; the current fixed guard does not achieve that. Human perceptual quality was not measured.',
    '', '**Decision for the next investment:** NO to claiming a validated method or starting a full-paper write-up from these results. YES to a narrowly scoped research investigation of the measured tradeoff, if that is the intended next stage. The pilot is complete; a complete pilot does not imply a positive hypothesis or completed research contribution.','',
    '## Artifacts and reproduction','',
    '- `agent_pilot_protocol.json`, `ami_agent_protocol.json`: fixed protocols and explicit AMI input-only amendment.',
    '- `results/agent_pilot_measurements.json`: all 40 primary evaluations, per-frame scores, candidate quality scores, hashes and selectors.',
    '- `results/ami_agent_pilot.json`: all six meeting-domain evaluations and source metadata.',
    '- `results/qwen_agent_pilot.json`: actual API agent traces, no secret.',
    '- `results/agent_pilot_audit.json`: independent arithmetic/provenance/label-isolation checks.',
    '- Run scripts in order: `prepare_agent_pilot.py`, `agent_pilot_measure.py`, `ami_agent_pilot.py`, `qwen_evidence_agent.py`, `audit_agent_pilot.py`, `report_agent_pilot.py`. Qwen reads `GROQ_API_KEY`; optionally choose an account-available Qwen ID using `GROQ_QWEN_MODEL`. Existing measurement caches require matching protocol/code/model hashes.','',
    'Sources: [UniTalk-ASD pinned dataset](https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD/tree/0c0f80331c91b57ce772bc674f81d40a24247383), [FastDVDnet](https://github.com/m-tassano/fastdvdnet), [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD), [AMI signals and camera mapping](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml), [AMI known data issues](https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml), [Groq supported models](https://console.groq.com/docs/models).','',
    'Novelty is not established by this experiment. Generic metric failures, conservative selection, input consistency and agent tool use are existing ideas. No first-of-its-kind claim is made.']
    (ROOT/'QWEN_AGENT_PILOT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Inspectable scientific evidence panel: actual pixels, not generated illustrations.
    case=next(c for c in r['cases'] if c['source']=='EEey6Pt0rqI' and c['label']==0 and c['sigma']==40)
    a=np.load(case['outputs']);canvas=Image.new('RGB',(768,300),'white');d=ImageDraw.Draw(canvas)
    for j,n in enumerate(['reference','captured','fastdvdnet']):
        canvas.paste(Image.fromarray(a[n][12]).resize((256,256)),(j*256,40));d.text((j*256+5,4),n,fill='black');d.text((j*256+5,20),f"TalkNet error: {case['candidates'][n]['error']*100:.2f}%",fill='black')
    canvas.save(ROOT/'results/agent_pilot_failure.png')
    print('Report written; Qwen completed cases:',len(completed))
if __name__=='__main__':main()
