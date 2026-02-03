# Instructor Guide

**Suggested run-of-show (90 minutes)**
1. Intro and goals. Use slides 1 to 2 to frame the journey.
2. DPO concept. Use slides 3 to 5 and highlight preference pairs.
3. DPO hands-on. Students complete the DPO TODOs and run training.
4. GRPO concept. Use slides 6 to 8 and emphasize reward signals.
5. GRPO hands-on. Students complete the GRPO TODOs and run training.
6. Compare models in the chat app and discuss results.
7. Risks and safety. Use slides 9 to 12 for discussion.

**Common pitfalls to watch**
- Missing `WANDB_API_KEY` or Hugging Face login.
- GPU not available or incompatible PyTorch build.
- Running out of VRAM when using larger models.

**Discussion prompts**
- Why does DPO work without an explicit reward model?
- When would GRPO be worth the extra compute cost?
- What kinds of reward hacking could appear here?
