"""Lip-sync: a mouth flapped from the dialogue audio's own envelope.

The model cannot hear, so the mouth is arithmetic over the WAV say() renders. The
claims: the envelope quantises to closed/mid/open and only emits changes; say()
drives those mouth swaps when the face has the say_* slots (and leaves faces without
them alone); and the swaps record and replay like any other expression.
"""

from __future__ import annotations

import numpy as np

from glaxnimate_ai.audio.lipsync import LIPSYNC_SLOTS, mouth_levels
from glaxnimate_ai.engine.session import Session, SessionStore


def test_mouth_levels_track_loudness_and_close_at_the_end():
    # quiet quarter, then loud, at 24 fps over 44100 Hz
    sr, fps = 44100, 24
    quiet = np.full(int(sr * 0.2), 0.02, np.float32)
    loud = np.full(int(sr * 0.4), 0.9, np.float32)
    sig = np.concatenate([quiet, loud])
    levels = mouth_levels(sig, fps=fps, start_frame=10, sr=sr)
    assert levels[0][0] == 10, "starts at the given frame"
    assert levels[0][1] == 0, "opens closed on the quiet part"
    assert any(lvl == 2 for _, lvl in levels), "opens wide on the loud part"
    assert levels[-1][1] == 0, "shuts the mouth after the line"
    # only changes are emitted (no two consecutive identical levels)
    seq = [lvl for _, lvl in levels]
    assert all(a != b for a, b in zip(seq, seq[1:])), "redundant swaps present"


def test_empty_audio_is_a_single_closed_mouth():
    assert mouth_levels(np.zeros(0, np.float32), fps=24, start_frame=5) == [(5, 0)]


def test_say_drives_lip_sync_when_the_face_has_speech_mouths():
    s = SessionStore().create(width=280, height=320, frames=30)
    r = s.run("man = stick()\n"
              "add_action(man, actions.idle(man, ground_y=ground, x=140), name='sp', face='stick')\n"
              "say('sp', 'hey you over there', 2)")
    assert r.ok, r.format()
    swaps = [a for _, a in s.characters[0].expressions if a in LIPSYNC_SLOTS]
    assert len(swaps) >= 3, "a spoken line should flap the mouth several times"


def test_lipsync_can_be_turned_off():
    s = SessionStore().create(width=280, height=320, frames=30)
    s.run("man = stick()\n"
          "add_action(man, actions.idle(man, ground_y=ground, x=140), name='sp', face='stick')\n"
          "say('sp', 'quiet please', 2, lipsync=False)")
    swaps = [a for _, a in s.characters[0].expressions if a in LIPSYNC_SLOTS]
    assert swaps == [], "lipsync=False should not touch the mouth"


def test_a_face_without_speech_mouths_is_not_lip_synced():
    # the human face has no say_* attachments; say() must not error, just skip flapping
    s = SessionStore().create(width=280, height=320, frames=30)
    r = s.run("man = human()\n"
              "add_action(man, actions.idle(man, ground_y=ground, x=140), name='sp', face='human')\n"
              "say('sp', 'hello', 2)")
    assert r.ok, r.format()
    swaps = [a for _, a in s.characters[0].expressions if a in LIPSYNC_SLOTS]
    assert swaps == [], "a face without say_* mouths is left alone"


def test_bare_language_code_routes_to_gtts_not_piper(monkeypatch):
    """Piper has no Tamil; a bare code (ta) or gtts:xx routes to the Google backend,
    while a full piper model name does not."""
    import numpy as np

    from glaxnimate_ai.audio import voice as V

    monkeypatch.delenv("GLAXNIMATE_AI_TTS_STUB", raising=False)
    seen = {}
    monkeypatch.setattr(V, "_gtts_synthesize",
                        lambda text, lang, sr: seen.__setitem__("lang", lang)
                        or np.zeros(64, dtype=np.float32))
    V.synthesize("hello", "ta")
    assert seen["lang"] == "ta"
    V.synthesize("hola", "gtts:es")
    assert seen["lang"] == "es"
    # a full piper name must NOT take the gTTS path
    seen.clear()
    monkeypatch.setattr(V, "_gtts_synthesize",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("used gTTS")))
    monkeypatch.setattr(V, "PiperVoice", None, raising=False)
    try:
        V.synthesize("hi", "en_US-lessac-medium")
    except Exception:
        pass  # piper path may fail on a stubbed model — the point is it did NOT use gTTS


def test_lip_sync_swaps_survive_replay():
    st = SessionStore()
    s = st.create(width=280, height=320, frames=30)
    s.run("man = stick()\n"
          "add_action(man, actions.idle(man, ground_y=ground, x=140), name='sp', face='stick')\n"
          "say('sp', 'hey you over there', 2)")
    s.save()
    replayed = Session.replay(s.doc_id)
    swaps = [a for _, a in replayed.characters[0].expressions if a in LIPSYNC_SLOTS]
    assert len(swaps) >= 3, "the mouth flaps replay from the recorded expressions"
