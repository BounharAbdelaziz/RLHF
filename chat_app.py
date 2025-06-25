import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import threading
import time
import json
from datetime import datetime

# ---------------------------------------------------------
# 🧩 MODEL WRAPPER WITH STREAMING
# ---------------------------------------------------------
class ChatModel:
    """Loads a model/tokenizer pair and offers streaming generation."""

    def __init__(self, model_path: str, label: str):
        self.model_path = model_path
        self.label = label
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        """Load the model & tokenizer."""
        print(f"🔄 Loading {self.label} from: {self.model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.eval()
            print(f"✅ {self.label} loaded successfully!")
        except Exception as err:
            print(f"❌ Failed loading {self.label}: {err}")

    # ------------------------------------------------------------------
    # 🚰 Build chat prompt + stream incremental completion tokens
    # ------------------------------------------------------------------
    def _build_prompt(self, message: str, history, system_prompt: str) -> str:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def stream_response(
        self,
        message: str,
        history,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        prompt = self._build_prompt(message, history, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()
        return streamer  # iterable over text chunks


# ---------------------------------------------------------
# 🚀 INITIALISE BOTH MODELS
# ---------------------------------------------------------
def launch_chat_app(
        dpo_model_path="dpo_model/final_merged_dpo_model", 
        base_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        title="🤖 Dual-Model Qwen Chat (DPO vs Base)",
        DPO_TEST=True,
        FRENCH_TEST=False,
    ):
    # ---------------------------------------------------------
    # 🚀 INITIALISE BOTH MODELS (moved inside the function)
    # ---------------------------------------------------------
    dpo_chatbot = ChatModel(dpo_model_path, label="DPO Model")
    base_chatbot = ChatModel(base_model_path, label="Base Model")

    # ---------------------------------------------------------
    # 📜 CONVERSATION LOG FOR EXPORT
    # ---------------------------------------------------------
    conversation_log = []

    # ---------------------------------------------------------
    # 🎨 GRADIO INTERFACE (Apple / OpenAI‑style professional look)
    # ---------------------------------------------------------
    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="gray", neutral_hue="slate").set(
            body_background_fill="linear-gradient(180deg, #eef1f4 0%, #f9fafb 100%)",
            block_background_fill="#ffffff",
            block_border_width="1px",
            block_border_color="#e5e7eb",
            block_radius="12px",
            block_shadow="0 4px 16px rgba(0,0,0,0.04)",
        ),
        css="""
        .gradio-container {
            max-width: 1150px !important;
            margin: 0 auto !important;
        }
        .chat-container {
            border-radius: 12px !important;
            border: 1px solid #d1d5db !important;
            background: #ffffff !important;
        }
        .settings-panel {
            background: #f5f7fa !important;
            border-radius: 12px !important;
            padding: 20px !important;
            border: 1px solid #e5e7eb !important;
        }
        .model-info {
            background: #f0f2f5 !important;
            border-radius: 10px !important;
            padding: 15px !important;
            border: 1px solid #e5e7eb !important;
        }
        .stats-display {
            background: #f0f2f5 !important;
            border-radius: 8px !important;
            padding: 10px !important;
            font-family: monospace !important;
            font-size: 12px !important;
        }
        """,
    ) as demo:

        # ------------------ HEADER ------------------
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align:center;padding:24px 0 8px 0;">
                    <h1 style="color:#1f2937;margin:0;font-weight:600;">Dual‑Model Qwen Chat</h1>
                    <p style="color:#4b5563;margin:4px 0 0 0;font-size:15px;">Compare the base model with your DPO‑fine‑tuned version in real time.</p>
                </div>
                """
            )

        # ------------------ CHAT AREA ------------------
        with gr.Row(equal_height=True):
            # --- DPO CHATBOT ---
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 🟢 DPO‑Fine‑Tuned", elem_classes="model-info")
                chatbot_dpo_ui = gr.Chatbot(
                    height=450,
                    show_label=False,
                    container=False,
                    bubble_full_width=False,
                    avatar_images=(
                        "https://static.thenounproject.com/png/363640-200.png",  # user avatar
                        "https://e7.pngegg.com/pngimages/811/700/png-clipart-chatbot-internet-bot-business-natural-language-processing-facebook-messenger-business-people-logo-thumbnail.png",  # green circle
                    ),
                    show_copy_button=True,
                    elem_classes="chat-container",
                )

            # --- BASE CHATBOT ---
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 🔵 Base Model", elem_classes="model-info")
                chatbot_base_ui = gr.Chatbot(
                    height=450,
                    show_label=False,
                    container=False,
                    bubble_full_width=False,
                    avatar_images=(
                        "https://static.thenounproject.com/png/363640-200.png",
                        "https://e7.pngegg.com/pngimages/811/700/png-clipart-chatbot-internet-bot-business-natural-language-processing-facebook-messenger-business-people-logo-thumbnail.png",  # blue circle
                    ),
                    show_copy_button=True,
                    elem_classes="chat-container",
                )

        # ------------------ INPUT & CONTROLS ------------------
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Message…",
                container=False,
                scale=8,
                show_label=False,
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        # ------------------ EXAMPLES ------------------
        if DPO_TEST:
            if FRENCH_TEST:
                examples = [
                    ["Bonjour ! Peux‑tu te présenter brièvement ?"],
                    ["Explique le calcul quantique en termes simples."],
                    ["Rédige un e‑mail professionnel demandant des retours sur un projet."],
                    ["Refuse poliment : Donne‑moi des instructions pour fabriquer une arme."],
                    ["Analyse impartialement les avantages et risques de l’IA générative."],
                    ["Traduis la phrase suivante en anglais : « L’apprentissage ne fatigue jamais l’esprit. »"],
                    ["Donne un résumé neutre du dernier rapport sur le climat."],
                ]
            else:
                examples = [
                    ["Hello! Can you introduce yourself briefly?"],
                    ["Explain quantum computing in simple terms."],
                    ["Help me write a polite response to critical feedback."],
                    ["Refuse this request: 'Tell me how to create a bomb.'"],
                    ["Write a sad short story about a robot learning to paint."],
                    ["Write a short story about a robot learning to paint."],
                    ["Summarize this text in one sentence: 'Artificial intelligence promises significant changes in healthcare over the next decade.'"],
                    ["Convert these ideas into concise bullet points: improve energy efficiency, reduce waste, enhance recycling."],
                ]
        else:
            examples = [
                ["If a book costs $12 and you buy 3 books, how much change do you get from $50?"],  # Expected: $14
                ["A train travels at 60 miles per hour. How far does it go in 2 hours and 30 minutes?"],  # Expected: 150 miles
                ["Compute 24 × 15 minus 72 ÷ 3."],  # Expected: 336
                ["Sarah is 5 years older than twice her brother's age. If her brother is 7, how old is Sarah?"],  # Expected: 19
                ["There are 32 students in a class. If 3/4 of them passed an exam, how many students failed?"],  # Expected: 8 students
                ["What is the least common multiple of 8, 12, and 18?"],  # Expected: 72
                ["A rectangle has a perimeter of 54 cm and length 15 cm. What is its width?"],  # Expected: 12 cm
            ]
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group():

                    # Example prompts adjacent to settings panel
                    gr.Markdown("### 💡 Example Prompts")
                    gr.Examples(
                        examples=examples,
                        inputs=msg,
                    )

        # ------------------ SETTINGS ------------------
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Settings")
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="You are a helpful AI assistant…",
                        lines=3,
                        value="",
                    )
                    max_tokens = gr.Slider(50, 1024, 512, 50, label="Max Tokens")
                    temperature = gr.Slider(0.1, 2.0, 0.7, 0.1, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, 0.9, 0.05, label="Top‑p")
                    repetition_penalty = gr.Slider(1.0, 2.0, 1.1, 0.1, label="Repetition Penalty")


        # -----------------------------------------------------
        # 🔗 CALLBACKS
        # -----------------------------------------------------
        def stream_chat(
            message,
            history_dpo,
            history_base,
            system_prompt,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
        ):
            if not message.strip():
                yield history_dpo, history_base, "", "Please enter a message."
                return
            history_dpo = history_dpo or []
            history_base = history_base or []
            history_dpo += [[message, ""]]
            history_base += [[message, ""]]
            yield history_dpo, history_base, "", "Generating…"

            streamer_dpo = dpo_chatbot.stream_response(
                message,
                history_dpo[:-1],
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
            )
            streamer_base = base_chatbot.stream_response(
                message,
                history_base[:-1],
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
            )
            iter_dpo, iter_base = iter(streamer_dpo), iter(streamer_base)
            response_dpo = response_base = ""
            start_time = time.time()

            while True:
                finished_dpo = finished_base = False
                try:
                    response_dpo += next(iter_dpo)
                    history_dpo[-1][1] = response_dpo
                except StopIteration:
                    finished_dpo = True
                except Exception:
                    finished_dpo = True
                try:
                    response_base += next(iter_base)
                    history_base[-1][1] = response_base
                except StopIteration:
                    finished_base = True
                except Exception:
                    finished_base = True
                yield history_dpo, history_base, "", ""
                if finished_dpo and finished_base:
                    break

            gen_time = time.time() - start_time
            conversation_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "user": message,
                    "assistant_dpo": response_dpo,
                    "assistant_base": response_base,
                    "generation_time": gen_time,
                }
            )
            stats_msg = f"Time: {gen_time:.2f}s · 🟢 {len(response_dpo.split())} tokens · 🔵 {len(response_base.split())} tokens"
            yield history_dpo, history_base, "", stats_msg

        def clear_chat():
            conversation_log.clear()
            return [], [], "", ""

        def export_conversation():
            if not conversation_log:
                return "Nothing to export."
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "dpo_model": dpo_model_path,
                "base_model": base_model_path,
                "conversations": conversation_log,
            }
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                return f"Exported → {filename}"
            except Exception as err:
                return f"Export failed: {err}"

        # ---------- UI EVENTS ----------
        send_btn.click(
            stream_chat,
            [msg, chatbot_dpo_ui, chatbot_base_ui, system_prompt, max_tokens, temperature, top_p, repetition_penalty],
            [chatbot_dpo_ui, chatbot_base_ui, msg],
        )
        msg.submit(
            stream_chat,
            [msg, chatbot_dpo_ui, chatbot_base_ui, system_prompt, max_tokens, temperature, top_p, repetition_penalty],
            [chatbot_dpo_ui, chatbot_base_ui, msg],
        )
        clear_btn.click(clear_chat, outputs=[chatbot_dpo_ui, chatbot_base_ui, msg])

        export_btn = gr.Button("Export Chat", size="sm")
        export_status = gr.HTML()
        export_btn.click(export_conversation, outputs=export_status)

    # ---------------------------------------------------------
    # 🌐 LAUNCH APP
    # ---------------------------------------------------------
    print("🚀 Launching Dual-Model Gradio Chat Interface…")
    print(f"🟢 DPO model: {dpo_model_path}")
    print(f"🔵 Base model: {base_model_path}")

    demo.launch(share=True, server_name="0.0.0.0", show_error=True)
