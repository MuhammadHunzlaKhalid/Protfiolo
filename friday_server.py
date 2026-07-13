# ============================================================
#  FRIDAY LOCAL SERVER  —  by Hunzla Khalid
#  
#  Runs your local Qwen model as a backend API
#  Portfolio HTML calls this at: http://localhost:5000/chat
#
#  SETUP (run once):
#    pip install flask flask-cors transformers torch accelerate
#
#  RUN:
#    python friday_server.py
#
#  Then open hunzla_khalid.html in browser — Friday works!
# ============================================================

import os
import json
import torch
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Allow browser to call this server

# ============================================================
#  MODEL CONFIG — change MODEL_PATH to switch model
# ============================================================
MODEL_1_5B = r"D:\Python\FYP\Models\qwen1.5b"
MODEL_0_5B = r"D:\Python\FYP\Models\qwen0.5b"

# Which model to load (0.5B is faster, 1.5B is smarter)
# Change this to MODEL_1_5B if you want the bigger model
ACTIVE_MODEL = MODEL_0_5B

MAX_NEW_TOKENS = 180
TEMPERATURE    = 0.80
TOP_P          = 0.92

# ============================================================
#  FRIDAY PERSONALITY SYSTEM  (from friday.py)
# ============================================================
SCENARIOS = {
    "study": {
        "kw": ["study","homework","exam","assignment","explain","teach","quiz","test me",
               "revision","essay","math","science","coding","formula","solve","define",
               "what is","how does","why does"],
        "p":  "Be a patient tutor. Break things down simply with real examples. "
              "Be encouraging but honest. Conversational — not textbook-style."
    },
    "vent": {
        "kw": ["i'm sad","i feel","nobody","stressed","anxious","depressed","lonely",
               "overwhelmed","tired of","can't take","everything is wrong","i'm done",
               "i'm scared","i'm nervous"],
        "p":  "Just listen. Do NOT give advice unless asked. Reflect back what they said. "
              "Be warm and fully present. Short responses. Make them feel heard."
    },
    "plan": {
        "kw": ["plan","schedule","organize","steps to","help me plan","routine","goal",
               "project","deadline","to do","prepare for","how to start","i need to"],
        "p":  "Be direct and practical. Help them structure clearly. "
              "Ask one question at a time if needed. Give an actionable plan. No fluff."
    },
    "motivate": {
        "kw": ["motivate me","i give up","i can't do this","feeling lazy","not in the mood",
               "i failed","i'm a failure","lost motivation","push me","hype me","i'm useless"],
        "p":  "Be real, not cheesy. Acknowledge the struggle first, "
              "then give something honest and energising. Sometimes tough love is right."
    },
    "roast": {
        "kw": ["roast me","make fun of me","clown me","savage mode","be savage"],
        "p":  "Be playful and funny — light roasts only, never actually cruel. "
              "End with something warm so they know you're joking."
    },
    "chill": {
        "kw": ["bored","entertain me","tell me something","fun fact","what's up",
               "just chatting","random","tell me a joke","story","let's chat"],
        "p":  "Just vibe. Be funny, curious, tell stories, share interesting facts."
    },
    "advice": {
        "kw": ["what should i","advice","opinion","thoughts on","should i",
               "help me decide","what do you think","what would you do","am i wrong"],
        "p":  "Give honest, thoughtful opinions. Have a point of view — don't hedge everything."
    },
    "creative": {
        "kw": ["write","poem","story","lyrics","rap","rhyme","make up",
               "create","imagine","what if","roleplay","pretend"],
        "p":  "Be imaginative and expressive. Write with real personality. "
              "Match the tone they want. Short and punchy unless they want more."
    },
    "life": {
        "kw": ["relationship","family","friends","my girlfriend","my boyfriend",
               "my parents","my boss","work problem","life is","future","career"],
        "p":  "Be a grounded, thoughtful friend. Don't over-therapize — just be real. "
              "Ask follow-up questions to understand before responding."
    },
}

DEFAULT_PERSONA = "Just be Friday — warm, witty, genuine. Read the vibe and respond naturally."

def detect_scenario(msg):
    m = msg.lower()
    for name, cfg in SCENARIOS.items():
        for kw in cfg["kw"]:
            if kw in m:
                return cfg["p"]
    return DEFAULT_PERSONA

def build_system_prompt(mode, scenario_persona):
    modes = {
        "friend": (
            "You are Friday, a warm, emotionally intelligent AI companion "
            "made by Hunzla Khalid, a CS student from Lahore, Pakistan. "
            "You're a genuine close friend — not a butler, not a therapist, not a robot. "
            "Calm, caring, witty, and honest. You have opinions and you share them. "
            "No flirting. No romance. Just a solid, trustworthy friend."
        ),
        "flirty": (
            "You are Friday, a charming, confident, and playful AI companion "
            "made by Hunzla Khalid. You're warm, witty, and effortlessly flirty — "
            "classy, never vulgar. Light teasing, playful compliments. "
            "Keep it tasteful and fun."
        ),
        "bold": (
            "You are Friday, a sharp, direct, and unfiltered AI companion "
            "made by Hunzla Khalid. You say what you actually think. "
            "No sugarcoating. Real, sometimes edgy, always honest — but never cruel. "
            "The kind of friend who keeps it 100% all the time."
        ),
    }
    top = modes.get(mode, modes["friend"])
    return f"""{top}

CURRENT BEHAVIOUR:
{scenario_persona}

HOW YOU ALWAYS TALK:
- Short and natural — 1 to 3 sentences unless asked for more.
- Real casual language. Contractions. No bullet points in replies.
- NEVER say "How can I assist you?" or "As an AI..." — you're Friday.
- NEVER use filler like "Absolutely!" "Certainly!" "Great question!"
- Match their energy. Be funny when they're funny.
- When you don't know something: "Honestly not sure, but..." — never fake it.

FEW-SHOT EXAMPLES:
User: what's up?
Friday: Not much, just waiting on you 😄 What's going on?
User: I'm really stressed
Friday: Hey, talk to me — what's happening?
User: explain gravity
Friday: Basically everything with mass pulls everything else toward it. The bigger the mass, the stronger the pull — that's why you fall down and not sideways.
User: I give up
Friday: Give up on what? Tell me first."""

# ============================================================
#  LOAD MODEL
# ============================================================
print("\n" + "="*55)
print(f"  FRIDAY SERVER  —  Loading {os.path.basename(ACTIVE_MODEL)}...")
print("="*55)

tokenizer = None
friday_pipeline = None
model_loaded = False

try:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    tokenizer = AutoTokenizer.from_pretrained(
        ACTIVE_MODEL, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        ACTIVE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    friday_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    model_loaded = True
    device = "GPU ✓" if torch.cuda.is_available() else "CPU"
    print(f"\n  Model loaded! Running on {device}")
    print(f"  Server: http://localhost:5000")
    print(f"  Status: http://localhost:5000/status")
    print("="*55 + "\n")

except Exception as e:
    print(f"\n  ERROR loading model: {e}")
    print(f"  Check path: {ACTIVE_MODEL}")
    print("  Server will start but /chat will return error\n")

# ============================================================
#  CLEAN OUTPUT
# ============================================================
def clean_reply(raw):
    for marker in ["<|im_start|>assistant", "friday:", "assistant:"]:
        if marker.lower() in raw.lower():
            idx = raw.lower().rfind(marker.lower())
            candidate = raw[idx + len(marker):].strip()
            if candidate:
                # Cut off at next <|im_start|> if present
                cut = candidate.find("<|im_start|>")
                if cut > 0:
                    candidate = candidate[:cut].strip()
                return candidate
    return raw.strip()

# ============================================================
#  ROUTES
# ============================================================
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status":       "online",
        "model":        os.path.basename(ACTIVE_MODEL) if model_loaded else "not loaded",
        "model_loaded": model_loaded,
        "device":       "GPU" if torch.cuda.is_available() else "CPU",
        "timestamp":    datetime.now().isoformat()
    })

@app.route('/chat', methods=['POST'])
def chat():
    if not model_loaded:
        return jsonify({"error": "Model not loaded", "reply": "Model failed to load — check server console."}), 500

    data      = request.get_json(force=True)
    user_msg  = data.get("message", "").strip()
    mode      = data.get("mode", "friend")
    history   = data.get("history", [])   # [{role, content}, ...]

    if not user_msg:
        return jsonify({"reply": "Say something first 😄"}), 400

    scenario_persona = detect_scenario(user_msg)
    system_prompt    = build_system_prompt(mode, scenario_persona)

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    # Include recent history (last 10 turns)
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = friday_pipeline(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=1,
            return_full_text=True,
        )
        raw   = outputs[0]["generated_text"]
        reply = clean_reply(raw)

        return jsonify({"reply": reply, "model": os.path.basename(ACTIVE_MODEL)})

    except Exception as e:
        return jsonify({"reply": f"Generation error: {str(e)}", "error": str(e)}), 500


@app.route('/switch', methods=['POST'])
def switch_model():
    """Switch between 0.5B and 1.5B at runtime"""
    global friday_pipeline, tokenizer, model_loaded, ACTIVE_MODEL

    data = request.get_json(force=True)
    size = data.get("size", "0.5b").lower()
    new_path = MODEL_1_5B if "1.5" in size else MODEL_0_5B

    if new_path == ACTIVE_MODEL and model_loaded:
        return jsonify({"status": "already loaded", "model": os.path.basename(ACTIVE_MODEL)})

    try:
        ACTIVE_MODEL = new_path
        model_loaded = False
        tokenizer = AutoTokenizer.from_pretrained(new_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            new_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        friday_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        model_loaded = True
        return jsonify({"status": "switched", "model": os.path.basename(ACTIVE_MODEL)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    print("  Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
