"""
AI COUNCIL - a four-agent round-table debate with live voice and synced captions.

  Researcher      (gpt-4.1-nano)   - opens the debate
  Domain Expert   (gpt-4.1-mini)   - practical industry view
  Critical Analyst(gpt-5-nano)     - challenges both
  Final Judge     (gpt-5-mini)     - delivers the verdict

How it stays fast
-----------------
* Pipelining: the moment a speaker's TEXT is ready, the next speaker starts
  writing while the current one's VOICE is still being synthesised and played.
  By the time a speaker finishes talking, the next one is usually ready.
* Optional speculative prefetch: while the Analyst is speaking, both possible
  next moves (Round-2 Researcher and the Round-1 verdict) are prepared, so the
  user's choice starts instantly. Turn off SPECULATIVE_PREFETCH to save cost.
* GPT-5 models run with minimal reasoning effort (big latency win for 70 words).
* Audio stays in memory (no mp3 files piling up on disk).
* One state machine drives the whole debate - no duplicate event handlers.

How the captions follow the voice
---------------------------------
The browser decodes the audio with the Web Audio API, so it knows the exact
duration and playback position. Words are revealed (dropping into place) as
playback advances, and the speaker's mouth moves with the real loudness of the
voice. When a speaker finishes, the browser asks the server for the next one.

Run:  pip install -U gradio openai python-dotenv
      echo OPENAI_API_KEY=sk-... > .env
      python ai_council.py
"""

from __future__ import annotations

import base64
import html
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

# Turn off Gradio's usage telemetry / update check (must be set before importing gradio).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

# Show only warnings and errors from libraries; keep our own app messages.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "httpcore", "openai", "gradio", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("ai_council")
log.setLevel(logging.INFO)

client = OpenAI(timeout=60, max_retries=2)  # reads OPENAI_API_KEY from env / .env

TTS_MODEL = "gpt-4o-mini-tts"
TURN_TIMEOUT = 120            # seconds to wait for one speaker (text + voice)
SPECULATIVE_PREFETCH = True   # prepare both next moves while the user decides
MAX_SESSIONS = 200            # in-memory sessions kept (oldest are dropped)

POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="council")

GPT41 = {"temperature": 0.8, "max_output_tokens": 220}
GPT5 = {"reasoning": {"effort": "minimal"}, "text": {"verbosity": "low"}, "max_output_tokens": 800}


# =========================================================
# AGENTS  (look = how the animated human is drawn)
# =========================================================

@dataclass(frozen=True)
class Agent:
    key: str
    role: str
    model_label: str
    api_model: str
    llm: dict
    voice: str
    tts_style: str
    color: str
    waiting: str
    look: dict


AGENTS = {
    "researcher": Agent(
        key="researcher", role="Researcher",
        model_label="GPT-4.1 NANO", api_model="gpt-4.1-nano", llm=GPT41,
        voice="echo",
        tts_style="Speak like a curious, articulate researcher opening a debate: clear, measured, lightly energetic.",
        color="#3ba7ff", waiting="Waiting to open the debate...",
        look=dict(skin="#e9b995", shade="#d29f7b", hair="#3a2a20", brow="#2f2219",
                  hair_style="short", outfit="#2b5ea8", lapel="#214b88", shirt="#eef2fa",
                  lips="#6b2a33", tie="#9fd3ff", extras=("glasses", "tie")),
    ),
    "expert": Agent(
        key="expert", role="Domain Expert",
        model_label="GPT-4.1 MINI", api_model="gpt-4.1-mini", llm=GPT41,
        voice="nova",
        tts_style="Speak like a confident, warm senior industry expert: practical, assured and conversational.",
        color="#32d6a0", waiting="Listening to the Researcher...",
        look=dict(skin="#cf9570", shade="#b67e5c", hair="#1e1310", brow="#1e1310",
                  hair_style="long", outfit="#1d8a69", lapel="#166e53", shirt="#f5efe6",
                  lips="#a8404f", extras=("earrings", "necklace", "blush")),
    ),
    "analyst": Agent(
        key="analyst", role="Critical Analyst",
        model_label="GPT-5 NANO", api_model="gpt-5-nano", llm=GPT5,
        voice="ash",
        tts_style="Speak like a sharp, skeptical analyst: calm and deliberate, with pointed emphasis on risks.",
        color="#ffb347", waiting="Taking notes on both sides...",
        look=dict(skin="#8f5b3c", shade="#784a2f", hair="#14100d", brow="#14100d",
                  hair_style="crop", outfit="#2b2f3b", lapel="#20232d", shirt="#e3e7ee",
                  lips="#b0646b", tie="#ffb347", beard="#1b1410", extras=("tie", "beard")),
    ),
    "judge": Agent(
        key="judge", role="Final Judge",
        model_label="GPT-5 MINI", api_model="gpt-5-mini", llm=GPT5,
        voice="onyx",
        tts_style="Speak like a wise presiding judge delivering a verdict: slow, authoritative and composed.",
        color="#b56cff", waiting="Observing the council...",
        look=dict(skin="#f1c8a7", shade="#dcae8b", hair="#cfd2dc", brow="#b9bcc7",
                  hair_style="receding", outfit="#15121f", lapel="#15121f", shirt="#f4f4f6",
                  lips="#6b2a33", beard="#d9dbe3", extras=("robe", "beard", "mustache")),
    ),
}


def who(agent_key: str) -> str:
    a = AGENTS[agent_key]
    return f"The {a.role}"


# =========================================================
# DEBATE STEPS
# =========================================================

SPEAKING_RULES = """
Your words will be read aloud at a round-table debate.
- Say between 40 and 70 words.
- Plain spoken sentences only: no headings, lists, markdown, emojis or stage directions.
- Do not restate the question and do not introduce yourself.
- When you respond to another council member, address them naturally by role.
"""


@dataclass(frozen=True)
class Step:
    key: str
    agent: str
    round: int            # 1, 2, or 3 (= verdict)
    label: str            # shown above the text in the card
    chip: str             # shown in the progress timeline
    responding_to: str
    sees: tuple           # earlier steps included in this speaker's transcript
    brief: str            # system instructions
    cue: str              # final line of the user message


_R1 = ("researcher", "expert", "analyst")

STEPS = {s.key: s for s in [
    Step("researcher", "researcher", 1, "ROUND 1 · OPENING", "Researcher", "", (),
         "You are the Researcher of the AI Council and you speak first. Open the debate with the most "
         "relevant facts, trends, opportunities, challenges and context. Do not reach a final "
         "conclusion; leave room for the other members to respond.",
         "Give your opening statement."),
    Step("expert", "expert", 1, "ROUND 1", "Expert", "Researcher", ("researcher",),
         "You are the Domain Expert of the AI Council. The Researcher has just opened. Respond to the "
         "Researcher from a practical, real-world industry perspective: acknowledge strong points, then "
         "add missing context, applications, limitations or corrections. Do not give the final conclusion.",
         "Respond directly to the Researcher."),
    Step("analyst", "analyst", 1, "ROUND 1", "Analyst", "Researcher + Domain Expert", _R1[:2],
         "You are the Critical Analyst of the AI Council. Challenge both the Researcher and the Domain "
         "Expert: expose weak assumptions, missing information, contradictions, risks and unrealistic "
         "expectations. Do not simply agree with anyone. Create useful disagreement they can answer. "
         "Do not give the final conclusion.",
         "Challenge both speakers."),
    Step("researcher_r2", "researcher", 2, "ROUND 2", "Researcher · R2", "Critical Analyst", _R1,
         "You are the Researcher of the AI Council, speaking in Round 2. Answer the Critical Analyst "
         "directly. Defend the strongest parts of your opening, and concede or correct what the challenge "
         "exposed. Do not defend blindly. Do not give the final conclusion.",
         "Give your Round 2 response to the Critical Analyst."),
    Step("expert_r2", "expert", 2, "ROUND 2", "Expert · R2", "Researcher + Critical Analyst",
         _R1 + ("researcher_r2",),
         "You are the Domain Expert of the AI Council, speaking in Round 2 - the last voice before the "
         "Final Judge. Address the Critical Analyst's concerns and the Researcher's Round 2 position, "
         "focusing on practical industry implications. Agree or disagree freely. Do not give the final "
         "council conclusion.",
         "Give your Round 2 response."),
    Step("judge_r1", "judge", 3, "VERDICT", "Verdict", "Entire council", _R1,
         "You are the Final Judge of the AI Council and the final speaker. Only Round 1 took place; the "
         "user chose to skip Round 2, so never pretend it happened. Weigh the discussion, name the "
         "strongest insight and the most important risk or limitation, and do not simply side with one "
         "speaker. End with a clear, decisive conclusion.",
         "Deliver the council's final judgment based only on Round 1."),
    Step("judge_r2", "judge", 3, "VERDICT", "Verdict", "Entire council",
         _R1 + ("researcher_r2", "expert_r2"),
         "You are the Final Judge of the AI Council and the final speaker. The council completed two "
         "rounds. Weigh the complete debate, name the strongest insights and the most important risks or "
         "limitations, and do not simply side with one speaker. End with a clear, decisive conclusion.",
         "Deliver the council's final judgment on the complete debate."),
]}

ROUND2_PATH = ("researcher_r2", "expert_r2", "judge_r2")
VERDICT_PATH = ("judge_r1",)

# Which steps appear in which seat's card
SEATS = {
    "researcher": ("researcher", "researcher_r2"),
    "expert": ("expert", "expert_r2"),
    "analyst": ("analyst",),
    "judge": ("judge_r1", "judge_r2"),
}


# =========================================================
# OPENAI CALLS
# =========================================================

_MARKDOWN = re.compile(r"[*_#`>]+")


def tidy(text: str) -> str:
    return " ".join(_MARKDOWN.sub("", text or "").split())


def write_turn(council: "Council", key: str) -> str:
    """Generate one speaker's statement from the transcript it is allowed to see."""
    step = STEPS[key]
    agent = AGENTS[step.agent]

    transcript = "\n\n".join(
        f"[{STEPS[k].label}] {AGENTS[STEPS[k].agent].role.upper()}: "
        f"{council.text_f[k].result(timeout=TURN_TIMEOUT)}"
        for k in step.sees
    ) or "(Nobody has spoken yet - you are opening the debate.)"

    response = client.responses.create(
        model=agent.api_model,
        instructions=step.brief + "\n" + SPEAKING_RULES,
        input=f"QUESTION:\n{council.question}\n\nDEBATE SO FAR:\n{transcript}\n\n{step.cue}",
        **agent.llm,
    )
    text = tidy(response.output_text)
    if not text:
        raise RuntimeError(f"{agent.role} returned an empty response")
    return text


_TTS_SLOTS = threading.BoundedSemaphore(2)   # avoid bursts of parallel voice requests


def synthesize(text: str, agent: Agent) -> str:
    """Text -> MP3 bytes (base64), kept in memory. Retries once on failure."""
    last_exc = None
    for attempt in range(2):
        try:
            with _TTS_SLOTS, client.audio.speech.with_streaming_response.create(
                model=TTS_MODEL,
                voice=agent.voice,
                input=text,
                instructions=agent.tts_style,
                response_format="mp3",
            ) as response:
                data = response.read()
            if len(data) < 1000:
                raise RuntimeError(f"voice API returned only {len(data)} bytes")
            return base64.b64encode(data).decode("ascii")
        except Exception as exc:
            last_exc = exc
            log.warning("Voice attempt %d failed for the %s: %s", attempt + 1, agent.role, exc)
    raise last_exc


# =========================================================
# COUNCIL SESSION  (state machine + background pipeline)
# =========================================================

class Council:
    def __init__(self, question: str, voice: bool):
        self.id = uuid.uuid4().hex
        self.question = question
        self.voice = voice
        self.text_f: dict[str, Future] = {}    # step -> statement text
        self.audio_f: dict[str, Future] = {}   # step -> base64 mp3 (or None)
        self.voice_error: dict[str, str] = {}  # step -> why its voice failed
        self.queue: list[str] = []             # steps still to be spoken
        self.spoken: list[str] = []            # steps already on stage
        self.current: str | None = None
        self.path: tuple | None = None         # ROUND2_PATH or VERDICT_PATH once chosen
        self.phase = "speaking"                # speaking | decision | done | error | stopped
        self.awaiting_next = False             # guards against duplicate "advance" clicks
        self.cancelled = False
        self.lock = threading.Lock()

    # ---- pipeline ------------------------------------------------------

    def prepare(self, keys, then=None):
        """Generate `keys` in order in the background; each voice is synthesised
        in parallel with the next speaker's writing."""
        with self.lock:
            fresh = [k for k in keys if k not in self.text_f]
            for k in fresh:
                self.text_f[k], self.audio_f[k] = Future(), Future()
        if fresh:
            POOL.submit(self._chain, fresh, then)
        elif then:
            then()

    def _chain(self, keys, then):
        for i, key in enumerate(keys):
            try:
                if self.cancelled:
                    raise RuntimeError("session stopped")
                text = write_turn(self, key)
            except Exception as exc:
                for k in keys[i:]:
                    for f in (self.text_f[k], self.audio_f[k]):
                        if not f.done():
                            f.set_exception(exc)
                return
            self.text_f[key].set_result(text)
            if self.voice:
                POOL.submit(self._voice, key, text)
            else:
                self.audio_f[key].set_result(None)
        if then and not self.cancelled:
            then()

    def _voice(self, key, text):
        try:
            audio = synthesize(text, AGENTS[STEPS[key].agent])
        except Exception as exc:
            log.error("Voice failed for %s - showing captions without sound: %s", key, exc)
            self.voice_error[key] = str(exc)[:200] or type(exc).__name__
            audio = None
        self.audio_f[key].set_result(audio)

    def speculate(self):
        self.prepare(VERDICT_PATH)
        self.prepare(ROUND2_PATH[:1])

    def cancel(self):
        self.cancelled = True
        self.phase = "stopped"

    def text(self, key) -> str:
        return self.text_f[key].result()


_SESSIONS: "OrderedDict[str, Council]" = OrderedDict()
_REGISTRY_LOCK = threading.Lock()


def register(council: Council):
    with _REGISTRY_LOCK:
        _SESSIONS[council.id] = council
        while len(_SESSIONS) > MAX_SESSIONS:
            _, old = _SESSIONS.popitem(last=False)
            old.cancel()


def lookup(sid) -> Council | None:
    return _SESSIONS.get(sid) if sid else None


# =========================================================
# ANIMATED HUMAN AVATARS  (inline SVG)
# =========================================================

HAIR_FRONT = {
    "short": "M33 52 C30 24 48 15 62 16 C81 17 91 31 87 52 C85 41 78 33 66 32 "
             "C57 36 45 36 39 41 C36 44 34 48 33 52 Z",
    "long": "M33 58 C28 24 46 17 62 17 C80 17 93 30 87 58 C85 45 80 36 73 31 "
            "C64 40 48 43 33 58 Z",
    "crop": "M35 45 C36 25 50 19 60 19 C72 19 84 25 85 45 C80 35 72 31 60 31 "
            "C48 31 40 35 35 45 Z",
    "receding": "M34 58 C31 44 34 35 41 31 L43 49 C40 51 37 54 34 58 Z "
                "M86 58 C89 44 86 35 79 31 L77 49 C80 51 83 54 86 58 Z",
}


def avatar_svg(agent: Agent, blink_delay: float) -> str:
    L = agent.look
    x = set(L.get("extras", ()))
    skin, shade, hair = L["skin"], L["shade"], L["hair"]
    p = [f'<svg class="avatar" viewBox="0 0 120 140" aria-hidden="true" '
         f'style="--blink-delay:{blink_delay}s">']

    # torso / clothing
    p.append(f'<path d="M12 140 C12 112 30 101 60 99 C90 101 108 112 108 140 Z" fill="{L["outfit"]}"/>')
    if "robe" in x:
        p.append(f'<path d="M30 106 L40 102 L46 140 L33 140 Z M90 106 L80 102 L74 140 L87 140 Z" '
                 f'fill="{agent.color}" opacity=".85"/>')
    if L["hair_style"] == "long":
        p.append(f'<path d="M30 54 C26 20 94 20 90 54 L94 104 C86 111 77 107 73 100 L47 100 '
                 f'C43 107 34 111 26 104 Z" fill="{hair}"/>')
    p.append(f'<path d="M50 78 L50 99 Q60 106 70 99 L70 78 Z" fill="{shade}"/>')
    if "robe" in x:
        p.append('<path d="M53 100 L67 100 L64 117 L60 123 L56 117 Z" fill="#f4f4f6"/>')
    else:
        p.append(f'<path d="M49 100 L60 118 L71 100 Q60 107 49 100 Z" fill="{L["shirt"]}"/>')
        if "tie" in x:
            p.append(f'<path d="M57.5 105 L62.5 105 L61.5 108 L63.5 121 L60 125 L56.5 121 L58.5 108 Z" '
                     f'fill="{L["tie"]}"/>')
        p.append(f'<path d="M49 100 L60 118 L53 128 L42 105 Z M71 100 L60 118 L67 128 L78 105 Z" '
                 f'fill="{L["lapel"]}"/>')
    if "necklace" in x:
        p.append('<path d="M51 101 Q60 111 69 101" fill="none" stroke="#f5c96b" stroke-width="1.3"/>'
                 '<circle cx="60" cy="108.5" r="1.9" fill="#f5c96b"/>')

    # head (animated group)
    p.append('<g class="head">')
    p.append(f'<ellipse cx="34" cy="57" rx="4.5" ry="7" fill="{shade}"/>'
             f'<ellipse cx="86" cy="57" rx="4.5" ry="7" fill="{shade}"/>')
    p.append(f'<ellipse cx="60" cy="54" rx="26" ry="30" fill="{skin}"/>')
    if "beard" in x:
        p.append(f'<path d="M34 58 C35 80 46 88 60 88 C74 88 85 80 86 58 C83 70 78 76 70 75 '
                 f'C64 71 56 71 50 75 C42 76 37 70 34 58 Z" fill="{L["beard"]}"/>')
    p.append(f'<path d="{HAIR_FRONT[L["hair_style"]]}" fill="{hair}"/>')
    if "blush" in x:
        p.append('<ellipse cx="44" cy="64" rx="4.5" ry="2.2" fill="#ff7f86" opacity=".28"/>'
                 '<ellipse cx="76" cy="64" rx="4.5" ry="2.2" fill="#ff7f86" opacity=".28"/>')
    p.append(f'<path d="M44 46 Q50 42.5 55 45.5 M65 45.5 Q70 42.5 76 46" stroke="{L["brow"]}" '
             f'stroke-width="2.2" fill="none" stroke-linecap="round"/>')
    p.append('<g class="eyes">'
             '<ellipse cx="50" cy="54" rx="2.8" ry="3.2" fill="#1b1b24"/>'
             '<ellipse cx="70" cy="54" rx="2.8" ry="3.2" fill="#1b1b24"/>'
             '<circle cx="51" cy="53" r=".9" fill="#fff"/><circle cx="71" cy="53" r=".9" fill="#fff"/>'
             '</g>')
    p.append(f'<path d="M60 56 Q57 63.5 59.5 65.5 Q61.5 66 63 65" stroke="{shade}" fill="none" '
             f'stroke-width="1.6" stroke-linecap="round"/>')
    if "mustache" in x:
        p.append(f'<path d="M50 69.5 C54 65 58 66 60 68 C62 66 66 65 70 69.5 C66 70.5 63 70.5 60 70 '
                 f'C57 70.5 54 70.5 50 69.5 Z" fill="{L["beard"]}"/>')
    p.append(f'<g class="mouth"><ellipse cx="60" cy="73" rx="6" ry="1.9" fill="{L["lips"]}"/></g>')
    if "glasses" in x:
        p.append('<g fill="none" stroke="#1d2433" stroke-width="1.8">'
                 '<circle cx="50" cy="54" r="7.2"/><circle cx="70" cy="54" r="7.2"/>'
                 '<path d="M57.2 54 Q60 52 62.8 54 M42.8 53 L35 51 M77.2 53 L85 51"/></g>')
    if "earrings" in x:
        p.append('<circle cx="34" cy="66" r="2.2" fill="#f5c96b"/><circle cx="86" cy="66" r="2.2" fill="#f5c96b"/>')
    p.append('</g></svg>')
    return "".join(p)


# Avatars never change, so build them once.
AVATAR_HTML = {k: avatar_svg(a, d) for (k, a), d in zip(AGENTS.items(), (0.0, 1.7, 3.1, 0.9))}


# =========================================================
# RENDERING
# =========================================================

esc = html.escape


def agent_card(c: Council | None, seat: str) -> str:
    agent = AGENTS[seat]
    spoken = [k for k in SEATS[seat] if c and k in c.spoken]
    speaking = bool(c and c.phase == "speaking" and c.current in SEATS[seat])

    if speaking:
        state, status = "active-agent", "SPEAKING"
    elif spoken:
        state, status = "finished-agent", "FINISHED"
    elif c and c.phase == "speaking" and c.queue and c.queue[0] in SEATS[seat]:
        state, status = "waiting-agent up-next", "UP NEXT"
    else:
        state, status = "waiting-agent", "WAITING"

    blocks = []
    for i, key in enumerate(spoken):
        step = STEPS[key]
        text = c.text(key)
        if speaking and key == c.current:
            words = " ".join(f'<span class="w">{esc(w)}</span>' for w in text.split())
            body = f'<div class="debate-text live" id="live-text">{words}</div>'
        else:
            body = f'<div class="debate-text">{esc(text)}</div>'
        divider = '<div class="round-divider"></div>' if i else ""
        blocks.append(f'{divider}<div class="round-label r{step.round}">{step.label}</div>{body}')
    content = "".join(blocks) or f'<div class="waiting-text">{agent.waiting}</div>'

    responding = ""
    if spoken and STEPS[spoken[-1]].responding_to:
        responding = (f'<div class="responding-to">RESPONDING TO '
                      f'<span>{esc(STEPS[spoken[-1]].responding_to)}</span></div>')

    return f"""
<div class="agent-card {state}" style="--agent-color:{agent.color}">
  <div class="portrait">{AVATAR_HTML[seat]}</div>
  <div class="voice-bars"><i></i><i></i><i></i><i></i><i></i></div>
  <div class="agent-role">{esc(agent.role)}</div>
  <div class="agent-model">{esc(agent.model_label)}</div>
  <div class="agent-status"><span class="status-dot"></span><span class="status-text">{status}</span></div>
  {responding}
  <div class="speech-box">{content}</div>
</div>"""


def stage_label(c: Council | None) -> str:
    if not c:
        return "COUNCIL READY"
    labels = {"decision": "ROUND 01 COMPLETE · YOUR CALL", "done": "SESSION ADJOURNED",
              "error": "SESSION INTERRUPTED", "stopped": "SESSION STOPPED"}
    if c.phase in labels:
        return labels[c.phase]
    key = c.current or (c.queue[0] if c.queue else None)
    if not key:
        return "CONVENING"
    return {1: "DEBATE · ROUND 01", 2: "DEBATE · ROUND 02", 3: "FINAL JUDGMENT"}[STEPS[key].round]


def timeline_html(c: Council | None) -> str:
    def chip(label, color, state):
        return f'<span class="chip {state}" style="--c:{color}">{esc(label)}</span>'

    def state_of(key):
        if not c:
            return ""
        if c.phase == "speaking" and c.current == key:
            return "active"
        return "done" if key in c.spoken else ""

    chips = [chip(STEPS[k].chip, AGENTS[STEPS[k].agent].color, state_of(k)) for k in _R1]
    decided = "active" if c and c.phase == "decision" else ("done" if c and c.path else "")
    chips.append(chip("Your call", "#e6e9ff", decided))
    if c and c.path:
        chips += [chip(STEPS[k].chip, AGENTS[STEPS[k].agent].color, state_of(k)) for k in c.path]
    else:
        chips.append(chip("Round 2 or Verdict", "#6b7394", ""))
    return '<div class="timeline">' + '<span class="chip-sep">›</span>'.join(chips) + "</div>"


def council_html(c: Council | None, status: str = "", thinking: bool = False, error: str = "") -> str:
    speaking = bool(c and c.phase == "speaking" and c.current)
    active_color = AGENTS[STEPS[c.current].agent].color if speaking else "#b56cff"
    question = c.question if c else "Waiting for a question..."
    status = status or ("Pose a question to convene the council" if not c else "")

    verdict = ""
    if c and c.phase == "done":
        judge_key = next((k for k in SEATS["judge"] if k in c.spoken), None)
        if judge_key:
            verdict = f"""
<div class="final-decision">
  <div class="final-title">⚖ FINAL COUNCIL DECISION</div>
  <div class="final-text">{esc(c.text(judge_key))}</div>
</div>"""

    error_html = f'<div class="council-error">{esc(error)}</div>' if error else ""

    return f"""
<div class="council-room {'is-speaking' if speaking else ''}" style="--active-color:{active_color}">
  <div class="council-header">
    <div class="main-title">AI COUNCIL</div>
    <div class="subtitle">MULTI-AGENT INTELLIGENCE CHAMBER</div>
    <div class="round-indicator">{stage_label(c)}</div>
  </div>

  <div class="question-panel">
    <div class="question-label">QUESTION BEFORE THE COUNCIL</div>
    <div class="question-text">{esc(question)}</div>
  </div>

  <div id="council-status" class="council-status {'thinking' if thinking else ''}">{esc(status)}</div>
  {timeline_html(c)}
  {error_html}

  <div class="judge-area">{agent_card(c, "judge")}<div class="judge-line"></div></div>

  <div class="table-area">
    <div class="researcher-position">{agent_card(c, "researcher")}</div>
    <div class="table-wrap">
      <div class="table-glow"></div>
      <div class="round-table">
        <div class="table-inner">
          <div class="table-emblem">⚖</div>
          <div class="table-center">AI COUNCIL</div>
          <div class="table-sub">{stage_label(c)}</div>
        </div>
      </div>
    </div>
    <div class="expert-position">{agent_card(c, "expert")}</div>
    <div class="analyst-position">{agent_card(c, "analyst")}</div>
  </div>
  {verdict}
</div>"""


def halt_bus() -> str:
    """Tells the browser to stop any audio that is still playing."""
    return f'<div data-halt="{uuid.uuid4().hex}"></div>'


def speech_bus(c: Council, key: str, audio_b64: str | None, next_msg: str) -> str:
    return (f'<div data-token="{c.id}:{key}" data-target="live-text" '
            f'data-next="{esc(next_msg, quote=True)}" '
            f'data-decision="{"1" if key == "analyst" else ""}" '
            f'data-voice-error="{esc(c.voice_error.get(key, ""), quote=True)}" '
            f'data-audio="{audio_b64 or ""}"></div>')


# =========================================================
# EVENT HANDLERS
# =========================================================

NOOP = (gr.update(), gr.update(), gr.update())   # (council view, audio bus, decision panel)


def step_outputs(c: Council | None):
    """Advance the council to its next speaker (or to the decision / final state)."""
    if not c:
        return NOOP
    with c.lock:
        if c.phase != "speaking" or not c.awaiting_next or c.cancelled:
            return NOOP
        c.awaiting_next = False
        if not c.queue:
            c.current = None
            c.phase = "decision" if c.spoken and c.spoken[-1] == "analyst" else "done"
            decision = c.phase == "decision"
            status = ("Round 1 is complete - continue the debate or call for the verdict"
                      if decision else "The council has reached its decision")
            return council_html(c, status=status), gr.update(), gr.update(visible=decision)
        key = c.queue.pop(0)

    try:
        c.text_f[key].result(timeout=TURN_TIMEOUT)
        audio = c.audio_f[key].result(timeout=TURN_TIMEOUT)
    except Exception as exc:
        if c.cancelled:
            return NOOP
        log.exception("Step %s failed", key)
        c.phase = "error"
        return (council_html(c, error=f"{who(STEPS[key].agent)} could not respond: {exc}"),
                gr.update(), gr.update(visible=False))

    with c.lock:
        if c.cancelled:
            return NOOP
        c.current = key
        c.spoken.append(key)
        c.awaiting_next = True
        upcoming = c.queue[0] if c.queue else None

    if upcoming:
        next_msg = f"{who(STEPS[upcoming].agent)} is preparing a response"
    elif key == "analyst":
        next_msg = "Round 1 complete"
    else:
        next_msg = "The judge has spoken"

    return (council_html(c, status=f"{who(STEPS[key].agent)} has the floor"),
            speech_bus(c, key, audio, next_msg),
            gr.update(visible=False))


def begin(question, voice, old_sid):
    old = lookup(old_sid)
    if old:
        old.cancel()

    question = (question or "").strip()
    if not question:
        yield "", council_html(None, status="Please enter a question for the council"), halt_bus(), gr.update(visible=False)
        return

    c = Council(question, bool(voice))
    register(c)
    c.queue = list(_R1)
    c.awaiting_next = True
    c.prepare(_R1, then=c.speculate if SPECULATIVE_PREFETCH else None)

    # Instant feedback, then the first speaker as soon as they're ready.
    yield (c.id,
           council_html(c, status=f"{who('researcher')} is preparing the opening statement", thinking=True),
           halt_bus(), gr.update(visible=False))
    yield (c.id, *step_outputs(c))


def advance(sid):
    return step_outputs(lookup(sid))


def choose(sid, path):
    c = lookup(sid)
    if not c:
        yield NOOP
        return
    with c.lock:
        if c.phase != "decision":
            yield NOOP
            return
        c.path = path
        c.queue = list(path)
        c.phase = "speaking"
        c.awaiting_next = True
    c.prepare(path)
    yield (council_html(c, status=f"{who(STEPS[path[0]].agent)} is preparing", thinking=True),
           gr.update(), gr.update(visible=False))
    yield step_outputs(c)


def choose_round2(sid):
    yield from choose(sid, ROUND2_PATH)


def choose_verdict(sid):
    yield from choose(sid, VERDICT_PATH)


def stop(sid):
    c = lookup(sid)
    if c:
        c.cancel()
    return council_html(c, status="Session stopped"), halt_bus(), gr.update(visible=False)


# =========================================================
# BROWSER SCRIPT - voice playback, synced captions, lip-sync
# =========================================================

COUNCIL_JS = r"""
() => {
  if (window.__council) return;
  const C = window.__council = { audio: null, url: null, raf: 0, watch: 0, token: null,
                                 unlocked: false, seen: new Set() };
  document.body.classList.add('dark', 'council-js');

  // One <audio> element is reused for every speaker. The browser unlocks it for
  // the whole session on the first click, so later speakers can play on their own.
  const SILENT = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=';
  const player = () => {
    if (!C.audio) {
      C.audio = new Audio();
      C.audio.preload = 'auto';
      C.audio.setAttribute('playsinline', '');
    }
    return C.audio;
  };
  const unlock = () => {
    if (C.unlocked || C.token) return;          // never interrupt a speaker
    const a = player();
    a.src = SILENT;
    const p = a.play();
    if (p) p.then(() => { C.unlocked = true; }).catch(() => {});
  };
  ['pointerdown', 'keydown', 'touchend'].forEach(t =>
    document.addEventListener(t, unlock, { capture: true, passive: true }));

  const toast = (msg) => {
    let t = document.getElementById('council-toast');
    if (!t) { t = document.createElement('div'); t.id = 'council-toast'; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 7000);
  };

  const removeTap = () => { const b = document.getElementById('tap-to-listen'); if (b) b.remove(); };

  const stop = () => {
    cancelAnimationFrame(C.raf); clearTimeout(C.watch);
    if (C.audio) { C.audio.onended = null; C.audio.onerror = null; C.audio.pause(); }
    removeTap();
    C.token = null;
  };

  const advance = () => {
    const el = document.getElementById('advance-btn');
    const btn = el && (el.tagName === 'BUTTON' ? el : el.querySelector('button'));
    if (btn) btn.click();
  };

  const showDecision = (tries = 0) => {
    const panel = document.querySelector('.decision-panel');
    if (panel && panel.offsetParent !== null) panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    else if (tries < 30) setTimeout(() => showDecision(tries + 1), 150);
  };

  // If the browser blocks sound, ask for one tap and start playback inside that tap.
  const tapToListen = (a) => new Promise((resolve, reject) => {
    removeTap();
    const b = document.createElement('button');
    b.id = 'tap-to-listen';
    b.textContent = '🔊  Tap to hear the council';
    b.onclick = () => { b.remove(); a.play().then(() => { C.unlocked = true; resolve(); }, reject); };
    document.body.appendChild(b);
  });

  // Loudness envelope for lip-sync. Decoded offline, so it never affects what you hear.
  async function envelope(blob) {
    const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!OAC) return null;
    const buf = await new OAC(1, 1, 22050).decodeAudioData(await blob.arrayBuffer());
    const data = buf.getChannelData(0), step = 0.03, n = Math.max(1, Math.floor(buf.sampleRate * step));
    const rms = [];
    for (let i = 0; i < data.length; i += n) {
      const end = Math.min(data.length, i + n); let s = 0, c = 0;
      for (let j = i; j < end; j += 4) { s += data[j] * data[j]; c++; }
      rms.push(Math.sqrt(s / Math.max(1, c)));
    }
    const peak = Math.max(0.05, ...rms);
    return { duration: buf.duration, step, rms: rms.map(v => v / peak) };
  }

  async function play(bus) {
    const token = bus.dataset.token;
    stop();
    C.token = token;
    const live = () => C.token === token;
    await new Promise(r => requestAnimationFrame(r));   // let the new council HTML land

    const target = document.getElementById(bus.dataset.target);
    const card = target ? target.closest('.agent-card') : null;
    const box = target ? target.closest('.speech-box') : null;
    const mouth = card ? card.querySelector('.mouth') : null;
    const words = target ? [...target.querySelectorAll('.w')] : [];
    if (target) {   // bring the start of the new passage into view if it is off-screen
      const r = target.getBoundingClientRect();
      if (r.top < 70 || r.top > innerHeight * 0.6) {
        window.scrollBy({ top: r.top - innerHeight * 0.3, behavior: 'smooth' });
      }
    }

    // Each word gets a share of the time proportional to its length plus punctuation pauses.
    let total = 0;
    const starts = words.map(w => {
      const s = total, t = w.textContent;
      total += t.length + 1.5 + (/[.!?]["')]?$/.test(t) ? 7 : /[,;:—-]$/.test(t) ? 3.5 : 0);
      return s;
    });
    // Keep the word being spoken on screen (only scrolls if it would go below the window).
    const follow = (w) => {
      const r = w.getBoundingClientRect();
      if (r.bottom > innerHeight - 24 && r.top < innerHeight + 200) {
        window.scrollBy({ top: r.bottom - innerHeight + 90, behavior: 'smooth' });
      }
    };
    let shown = 0;
    const reveal = p => {
      const limit = p * total;
      let moved = false;
      while (shown < words.length && starts[shown] <= limit) { words[shown++].classList.add('on'); moved = true; }
      if (moved) follow(words[shown - 1]);
    };

    const finish = () => {
      if (!live()) return;
      cancelAnimationFrame(C.raf); clearTimeout(C.watch);
      if (C.audio) { C.audio.onended = null; C.audio.onerror = null; }
      C.token = null;
      words.forEach(w => w.classList.add('on'));
      if (mouth) mouth.style.transform = '';
      if (card) {
        card.classList.remove('active-agent', 'lipsync');
        card.classList.add('finished-agent');
        const st = card.querySelector('.status-text'); if (st) st.textContent = 'FINISHED';
      }
      const status = document.getElementById('council-status');
      if (status && bus.dataset.next) { status.textContent = bus.dataset.next; status.classList.add('thinking'); }
      if (bus.dataset.decision) showDecision();
      advance();
    };

    if (bus.dataset.voiceError) toast('No voice for this speaker: ' + bus.dataset.voiceError);

    const b64 = bus.dataset.audio;
    if (b64) {
      try {
        const blob = await (await fetch('data:audio/mpeg;base64,' + b64)).blob();
        const env = await envelope(blob).catch(() => null);
        if (!live()) return;

        const a = player();
        if (C.url) URL.revokeObjectURL(C.url);
        C.url = URL.createObjectURL(blob);
        a.onended = finish;
        a.onerror = () => { if (live()) { toast('The browser could not play this voice clip.'); finish(); } };
        a.src = C.url;
        a.muted = false; a.volume = 1;
        try {
          await a.play();
        } catch (e) {
          if (e.name !== 'NotAllowedError' || !live()) throw e;
          await tapToListen(a);
        }
        if (!live()) { a.pause(); return; }
        C.unlocked = true;
        if (card && env) card.classList.add('lipsync');

        // Safety net: move on even if the browser never fires "ended".
        const dur = (env && env.duration) || a.duration;
        if (dur && isFinite(dur)) C.watch = setTimeout(finish, (dur + 4) * 1000);

        const tick = () => {
          if (!live()) return;
          const d = (env && env.duration) || a.duration;
          if (d && isFinite(d)) reveal(Math.min(1, a.currentTime / d));
          if (mouth && env) {
            const v = env.rms[Math.min(env.rms.length - 1, Math.floor(a.currentTime / env.step))] || 0;
            mouth.style.transform = `scaleY(${(1 + v * 3.2).toFixed(2)})`;
          }
          C.raf = requestAnimationFrame(tick);
        };
        tick();
        return;
      } catch (err) {
        if (!live()) return;
        console.warn('[AI Council] voice playback failed, using timed captions:', err);
        toast('Voice playback failed (' + (err.name || 'error') + '). Showing captions only.');
      }
    }

    // Silent mode (voice off or audio failed): reveal at a natural speaking pace.
    const duration = Math.max(2500, total * 55);
    const begin = performance.now();
    const tick = () => {
      if (!live()) return;
      const p = (performance.now() - begin) / duration;
      reveal(Math.min(1, p));
      if (p >= 1) finish(); else C.raf = requestAnimationFrame(tick);
    };
    tick();
  }

  const scan = () => {
    const halt = document.querySelector('#audio-bus [data-halt]');
    if (halt && !C.seen.has(halt.dataset.halt)) { C.seen.add(halt.dataset.halt); stop(); }
    const bus = document.querySelector('#audio-bus [data-token]');
    if (bus && !C.seen.has(bus.dataset.token)) { C.seen.add(bus.dataset.token); play(bus); }
  };
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
}
"""


# =========================================================
# CSS
# =========================================================

CSS = r"""
body, .gradio-container {
  background: radial-gradient(1200px 700px at 50% -10%, #1a2150 0%, #0a0d1f 45%, #04050b 100%) !important;
}
.gradio-container { max-width: 1480px !important; width: 100% !important; margin: auto !important; }
#advance-btn, #audio-bus { display: none !important; }

/* ---------- control deck ---------- */
.control-deck {
  border-radius: 20px !important; padding: 18px !important;
  background: rgba(10, 14, 32, .85) !important;
  border: 1px solid rgba(181, 108, 255, .25) !important;
}
.deck-title { font: 800 13px/1.4 system-ui, sans-serif; letter-spacing: 4px; color: #c8b6ff; }
.deck-sub { font: 13px/1.5 system-ui, sans-serif; color: #8f9abb; margin-top: 4px; }
.convene-btn { background: linear-gradient(135deg, #7b4dff, #3ba7ff) !important; color: #fff !important;
  border: none !important; letter-spacing: 2px; font-weight: 800 !important; }

/* ---------- room ---------- */
.council-room {
  --active-color: #b56cff;
  position: relative; width: 100%; box-sizing: border-box;
  padding: 34px 28px 44px; border-radius: 28px; overflow: hidden;
  color: #eef1ff; font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  background:
    radial-gradient(ellipse 60% 45% at 50% 58%, color-mix(in srgb, var(--active-color) 13%, transparent), transparent 70%),
    radial-gradient(ellipse 40% 30% at 50% 0%, rgba(255,255,255,.06), transparent 70%),
    linear-gradient(180deg, #0b0f24, #05060d);
  border: 1px solid rgba(255,255,255,.06);
  transition: background .6s ease;
}

.council-header { text-align: center; margin-bottom: 22px; }
.main-title {
  font-size: 44px; font-weight: 900; letter-spacing: 10px;
  background: linear-gradient(90deg, #cfe6ff, #ffffff, #e2c9ff);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  filter: drop-shadow(0 0 18px rgba(181,108,255,.45));
}
.subtitle { margin-top: 6px; font-size: 12px; letter-spacing: 5px; color: #8f9abb; }
.round-indicator {
  display: inline-block; margin-top: 14px; padding: 6px 16px; border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--active-color) 50%, transparent);
  background: color-mix(in srgb, var(--active-color) 12%, transparent);
  color: #e3d8ff; font-size: 10px; font-weight: 700; letter-spacing: 3px;
}

.question-panel {
  max-width: 900px; margin: 0 auto 18px; padding: 18px 26px; box-sizing: border-box;
  border-radius: 16px; text-align: center;
  background: rgba(10,15,35,.9); border: 1px solid rgba(181,108,255,.35);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.question-label { font-size: 10px; letter-spacing: 4px; color: #8f9abb; margin-bottom: 8px; }
.question-text { font-size: 20px; line-height: 1.5; overflow-wrap: anywhere; }

.council-status {
  text-align: center; min-height: 18px; margin: 0 auto 14px;
  font-size: 12px; letter-spacing: 1.5px; color: #aab3d6;
}
.council-status.thinking::before {
  content: "●"; margin-right: 8px; color: var(--active-color);
  animation: blink-dot 1s ease-in-out infinite;
}
@keyframes blink-dot { 50% { opacity: .2; } }

.timeline { display: flex; flex-wrap: wrap; justify-content: center; gap: 6px; max-width: 1000px; margin: 0 auto 30px; }
.chip {
  font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
  padding: 6px 11px; border-radius: 999px; color: #6f7899;
  border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.03);
}
.chip.done { color: #dfe6ff; border-color: color-mix(in srgb, var(--c) 55%, transparent); }
.chip.active {
  color: #fff; background: color-mix(in srgb, var(--c) 28%, transparent); border-color: var(--c);
  box-shadow: 0 0 16px color-mix(in srgb, var(--c) 50%, transparent);
}
.chip-sep { color: #3d4566; align-self: center; font-size: 12px; }

.council-error {
  max-width: 900px; margin: 0 auto 24px; padding: 14px 18px; border-radius: 12px;
  background: rgba(255,80,90,.1); border: 1px solid rgba(255,80,90,.45); color: #ffc9cd; font-size: 13px;
}

/* ---------- layout ---------- */
.judge-area { width: 330px; max-width: 100%; margin: 0 auto 60px; position: relative; z-index: 2; }
.judge-line {
  position: absolute; left: 50%; bottom: -60px; width: 2px; height: 60px; transform: translateX(-50%);
  background: linear-gradient(to bottom, #b56cff, transparent);
}
.table-area {
  max-width: 1250px; margin: 0 auto; display: grid;
  grid-template-columns: minmax(250px, 320px) minmax(300px, 1fr) minmax(250px, 320px);
  grid-template-areas: "researcher table expert" ". analyst .";
  gap: 26px 30px; align-items: center; justify-content: center;
}
.researcher-position { grid-area: researcher; }
.expert-position { grid-area: expert; }
.analyst-position { grid-area: analyst; justify-self: center; width: 100%; max-width: 360px; }

/* ---------- table ---------- */
.table-wrap { grid-area: table; position: relative; height: 290px; display: flex; align-items: center; justify-content: center; }
.round-table {
  position: relative; width: 100%; max-width: 580px; height: 100%; border-radius: 50%;
  background:
    radial-gradient(ellipse at 50% 30%, rgba(255,255,255,.08), transparent 55%),
    radial-gradient(ellipse at center, #232b52, #121733 55%, #090c1c);
  border: 2px solid color-mix(in srgb, var(--active-color) 55%, transparent);
  box-shadow: 0 0 40px color-mix(in srgb, var(--active-color) 25%, transparent),
              inset 0 0 60px rgba(59,167,255,.12), 0 30px 60px rgba(0,0,0,.55);
  display: flex; align-items: center; justify-content: center;
  transition: border-color .6s, box-shadow .6s;
}
.table-inner {
  width: 62%; height: 58%; border-radius: 50%; display: flex; flex-direction: column;
  align-items: center; justify-content: center; border: 1px dashed rgba(255,255,255,.1);
}
.is-speaking .round-table { animation: table-pulse 2.4s ease-in-out infinite; }
@keyframes table-pulse {
  50% { box-shadow: 0 0 70px color-mix(in srgb, var(--active-color) 45%, transparent),
                    inset 0 0 70px color-mix(in srgb, var(--active-color) 18%, transparent),
                    0 30px 60px rgba(0,0,0,.55); }
}
.table-emblem { font-size: 26px; color: color-mix(in srgb, var(--active-color) 70%, #fff); }
.table-center { font-size: 17px; letter-spacing: 6px; color: #8a95c2; font-weight: 800; margin-top: 4px; }
.table-sub { font-size: 9px; letter-spacing: 3px; color: #59628a; margin-top: 6px; }
.table-glow { position: absolute; inset: -22px; border-radius: 50%; border: 1px solid rgba(59,167,255,.14); }

/* ---------- agent card ---------- */
.agent-card {
  position: relative; box-sizing: border-box; padding: 18px 16px 16px; border-radius: 22px;
  background: linear-gradient(180deg, rgba(22,28,58,.94), rgba(9,12,26,.97));
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 12px 30px rgba(0,0,0,.4);
  transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease, opacity .35s ease;
}
.agent-card.waiting-agent { opacity: .7; }
.agent-card.up-next { opacity: .9; border-color: color-mix(in srgb, var(--agent-color) 30%, transparent); }
.agent-card.finished-agent { border-color: color-mix(in srgb, var(--agent-color) 35%, transparent); }
.agent-card.active-agent {
  opacity: 1; transform: translateY(-4px) scale(1.03); border-color: var(--agent-color);
  box-shadow: 0 0 0 1px var(--agent-color), 0 0 40px color-mix(in srgb, var(--agent-color) 45%, transparent);
}

.portrait {
  position: relative; width: 118px; height: 118px; margin: 0 auto; border-radius: 50%; overflow: hidden;
  background: radial-gradient(circle at 50% 30%, color-mix(in srgb, var(--agent-color) 35%, #1a2040), #0b0f22 72%);
  border: 3px solid color-mix(in srgb, var(--agent-color) 60%, transparent);
}
.portrait .avatar { position: absolute; left: 0; bottom: -2px; width: 100%; height: auto; }
.active-agent .portrait { animation: ring 1.6s ease-in-out infinite; }
@keyframes ring {
  0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--agent-color) 60%, transparent); }
  50% { box-shadow: 0 0 0 10px transparent; }
}

/* avatar motion */
.avatar .head { transform-box: view-box; transform-origin: 60px 96px; animation: idle-sway 7s ease-in-out infinite; }
.active-agent .avatar .head { animation: talk-sway 2.6s ease-in-out infinite; }
@keyframes idle-sway { 50% { transform: rotate(1.2deg); } }
@keyframes talk-sway {
  0%, 100% { transform: rotate(-2deg); }
  25% { transform: rotate(1.5deg) translateY(-1px); }
  50% { transform: rotate(2.5deg); }
  75% { transform: rotate(-1deg) translateY(-1.5px); }
}
.avatar .eyes { transform-box: fill-box; transform-origin: center; animation: blink 5.5s infinite; animation-delay: var(--blink-delay); }
@keyframes blink { 0%, 92%, 100% { transform: scaleY(1); } 95% { transform: scaleY(.08); } }
.avatar .mouth { transform-box: fill-box; transform-origin: center; }
.active-agent:not(.lipsync) .avatar .mouth { animation: mouth-talk .22s ease-in-out infinite alternate; }
@keyframes mouth-talk { from { transform: scaleY(1); } to { transform: scaleY(3.4); } }

.voice-bars { display: flex; gap: 3px; justify-content: center; align-items: flex-end; height: 14px; margin: 8px 0 4px; }
.voice-bars i { width: 3px; height: 3px; border-radius: 2px; background: var(--agent-color); opacity: .35; }
.active-agent .voice-bars i { opacity: 1; animation: bar .9s ease-in-out infinite; }
.active-agent .voice-bars i:nth-child(2) { animation-delay: .15s; }
.active-agent .voice-bars i:nth-child(3) { animation-delay: .3s; }
.active-agent .voice-bars i:nth-child(4) { animation-delay: .45s; }
.active-agent .voice-bars i:nth-child(5) { animation-delay: .6s; }
@keyframes bar { 50% { height: 14px; } }

.agent-role { text-align: center; font-size: 15px; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: var(--agent-color); }
.agent-model { text-align: center; font-size: 10px; letter-spacing: 2px; color: #7d87a8; margin-top: 4px; }
.agent-status { text-align: center; font-size: 10px; letter-spacing: 3px; margin: 8px 0 8px; color: var(--agent-color); }
.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--agent-color); margin-right: 6px; vertical-align: middle; }
.active-agent .status-dot { animation: blink-dot .8s ease-in-out infinite; }
.responding-to { text-align: center; font-size: 9px; letter-spacing: 2px; color: #6f7a9f; margin-bottom: 8px; }
.responding-to span { color: var(--agent-color); font-weight: 800; text-transform: uppercase; }

/* ---------- speech ---------- */
.speech-box {
  position: relative; min-height: 110px; box-sizing: border-box;
  padding: 14px; border-radius: 14px; background: rgba(3,5,12,.78);
  border: 1px solid rgba(255,255,255,.07); color: #e9ecff;
  font-size: 13.5px; line-height: 1.65; overflow-wrap: anywhere; scroll-behavior: smooth;
}
.active-agent .speech-box {
  border-color: color-mix(in srgb, var(--agent-color) 70%, transparent);
  box-shadow: inset 0 0 22px color-mix(in srgb, var(--agent-color) 10%, transparent);
}
.waiting-text { color: #6f7899; font-style: italic; }
.round-label { font-size: 9px; font-weight: 800; letter-spacing: 3px; color: #7f8aad; margin-bottom: 6px; }
.round-label.r2, .round-label.r3 { color: var(--agent-color); }
.round-divider { height: 1px; margin: 14px 0; background: rgba(255,255,255,.1); }

/* words drop into place as they are spoken */
.council-js .live .w {
  display: inline-block; opacity: 0; transform: translateY(-9px); filter: blur(2px);
  transition: opacity .3s ease, transform .3s ease, filter .3s ease;
}
.council-js .live .w.on { opacity: 1; transform: none; filter: none; }

#tap-to-listen {
  position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%); z-index: 9999;
  padding: 14px 26px; border-radius: 999px; cursor: pointer; border: none;
  background: linear-gradient(135deg, #7b4dff, #3ba7ff); color: #fff;
  font-size: 15px; font-weight: 800; letter-spacing: 1px;
  box-shadow: 0 10px 30px rgba(59,167,255,.45); animation: blink-dot 1.6s ease-in-out infinite;
}
#council-toast {
  position: fixed; right: 18px; bottom: 18px; z-index: 9999; max-width: min(420px, calc(100vw - 36px));
  padding: 12px 16px; border-radius: 12px; font: 13px/1.45 system-ui, sans-serif;
  background: rgba(40,14,24,.96); color: #ffd6db; border: 1px solid rgba(255,90,110,.5);
  opacity: 0; transform: translateY(10px); pointer-events: none; transition: opacity .3s, transform .3s;
}
#council-toast.show { opacity: 1; transform: none; }

/* ---------- verdict ---------- */
.final-decision {
  max-width: 900px; box-sizing: border-box; margin: 44px auto 0; padding: 26px 30px; border-radius: 20px;
  background: linear-gradient(180deg, rgba(34,18,62,.95), rgba(14,10,28,.97));
  border: 2px solid rgba(181,108,255,.6); box-shadow: 0 0 40px rgba(181,108,255,.22);
  animation: rise .6s ease both;
}
@keyframes rise { from { opacity: 0; transform: translateY(14px); } }
.final-title { text-align: center; color: #c79bff; font-size: 14px; font-weight: 800; letter-spacing: 4px; margin-bottom: 14px; }
.final-text { text-align: center; font-size: 16px; line-height: 1.75; color: #fff; }

/* ---------- decision panel (Gradio column) ---------- */
.decision-panel {
  max-width: 900px; margin: 24px auto !important; padding: 24px !important; box-sizing: border-box;
  text-align: center; border-radius: 20px !important;
  background: rgba(10,15,35,.96) !important; border: 1px solid rgba(181,108,255,.45) !important;
  box-shadow: 0 0 30px rgba(181,108,255,.15);
}
.decision-title { color: #fff; font-size: 16px; font-weight: 800; letter-spacing: 3px; margin-bottom: 8px; text-align: center; }
.decision-subtitle { color: #8f9abb; font-size: 13px; line-height: 1.5; margin-bottom: 8px; text-align: center; }
.decision-buttons { justify-content: center; gap: 14px; }
.decision-buttons button { min-width: 230px; font-weight: 800 !important; letter-spacing: 1.5px; }

/* ---------- responsive ---------- */
@media (max-width: 900px) {
  .council-room { padding: 18px 12px 28px; display: flex; flex-direction: column; }
  .judge-area { order: 5; margin: 26px auto 0 !important; }   /* judge speaks last, so sits last */
  .final-decision { order: 6; }
  .main-title { font-size: 30px; letter-spacing: 6px; }
  .subtitle { font-size: 9px; letter-spacing: 2px; }
  .question-text { font-size: 16px; }
  .table-area { display: flex; flex-direction: column; align-items: center; gap: 26px; }
  .table-wrap, .judge-line { display: none; }
  .researcher-position, .expert-position, .analyst-position, .judge-area { width: 100%; max-width: 420px; }
  .decision-buttons { flex-direction: column; align-items: center; }
}
@media (prefers-reduced-motion: reduce) {
  .avatar .head, .avatar .eyes, .active-agent .portrait, .is-speaking .round-table { animation: none !important; }
}
"""


# =========================================================
# GRADIO APP
# =========================================================

# Gradio 6 moved css/js from Blocks() to launch(); support both.
_GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
_ASSETS = {"css": CSS, "js": COUNCIL_JS}

with gr.Blocks(title="AI Council", **({} if _GRADIO_6 else _ASSETS)) as demo:
    session_id = gr.State("")

    with gr.Column(elem_classes=["control-deck"]):
        gr.HTML('<div class="deck-title">CONVENE THE AI COUNCIL</div>'
                '<div class="deck-sub">Four AI minds debate your question out loud, '
                'then a judge delivers the verdict.</div>')
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. Should small businesses in India adopt AI agents for customer support?",
            lines=2, max_lines=5,
        )
        with gr.Row():
            voice_toggle = gr.Checkbox(value=True, label="Voice narration", scale=1, min_width=160)
            start_button = gr.Button("CONVENE THE COUNCIL", variant="primary", scale=3,
                                     elem_classes=["convene-btn"])
            stop_button = gr.Button("STOP", variant="secondary", scale=1)

    council_view = gr.HTML(council_html(None))

    with gr.Column(visible=False, elem_classes=["decision-panel"]) as decision_panel:
        gr.HTML('<div class="decision-title">ROUND 1 COMPLETE</div>'
                '<div class="decision-subtitle">Should the council keep debating, '
                'or should the Final Judge rule now?</div>')
        with gr.Row(elem_classes=["decision-buttons"]):
            round2_button = gr.Button("CONTINUE TO ROUND 2", variant="primary")
            verdict_button = gr.Button("GO TO FINAL JUDGMENT", variant="secondary")

    # Invisible plumbing between the server and the browser script
    audio_bus = gr.HTML("", elem_id="audio-bus")
    advance_button = gr.Button("advance", elem_id="advance-btn")

    views = [council_view, audio_bus, decision_panel]

    for trigger in (start_button.click, question_box.submit):
        trigger(begin, [question_box, voice_toggle, session_id], [session_id, *views],
                show_progress="hidden")
    advance_button.click(advance, [session_id], views, show_progress="hidden")
    round2_button.click(choose_round2, [session_id], views, show_progress="hidden")
    verdict_button.click(choose_verdict, [session_id], views, show_progress="hidden")
    stop_button.click(stop, [session_id], views, show_progress="hidden")

demo.queue(default_concurrency_limit=32)

if __name__ == "__main__":
    demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
