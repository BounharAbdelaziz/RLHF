# RLHF Training Workshop (DPO + GRPO)

This repo supports a hands-on RLHF tutorial using Qwen2.5. It includes a student notebook with TODOs and a full solution notebook.

**What you will learn**
- DPO: align response style using preference pairs.
- GRPO: improve reasoning with reward-based practice.
- Compare a base model vs a tuned model in a live chat UI.

**Repo layout**
- `main` branch: `main.ipynb` is the full solution notebook.
- `student` branch: `main.ipynb` is the fill-in-the-blanks notebook.
- `student.ipynb` (on `main` branch) is the TODO version used to build the student branch.
- `chat_app.py` is the Gradio comparison UI.
- `utils.py` contains the LoRA merge helper.
- `presentation.pdf` and `presentation.pptx` are the slides.
- `docs/STUDENT.md` is the step-by-step student flow.
- `docs/INSTRUCTOR.md` is the facilitator run-of-show.

**Quickstart (local)**
1. `pip install -r requirements.txt`
2. `huggingface-cli login`
3. Set `WANDB_API_KEY` in your environment. You can copy `.env.example` to `.env` and export it in your shell.
4. Open `main.ipynb` (solution) or switch to the `student` branch for the TODO version.

**Quickstart (Colab)**
1. Upload or clone this repo in Colab.
2. Run the setup cells at the top of the notebook.

**Chat app**
- Run `python chat_app.py` after you have a merged DPO model at `dpo_model/final_merged_dpo_model`.

**Notes**
- This workshop assumes an NVIDIA GPU. Install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
- If you publish this repo, consider a `student` branch with only `student.ipynb` and a `main` branch with the solution.
