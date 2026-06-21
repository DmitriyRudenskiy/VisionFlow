from __future__ import annotations

import sys
import argparse


# ==================== БЫСТРАЯ ПРОВЕРКА АРГУМЕНТОВ ====================
def parse_args_early():
    parser = argparse.ArgumentParser(description="AI-анализатор аудио v6.6.1 (Final Hybrid)")
    parser.add_argument('files', nargs='*', help='Путь к аудиофайлу(ам)')
    parser.add_argument('--output', '-o', help='Папка для сохранения JSON')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять JSON файлы')
    parser.add_argument('--json', action='store_true', help='Выводить только сырой JSON в консоль')
    parser.add_argument('--no-color', action='store_true', help='Отключить цвета в консоли')
    parser.add_argument('--recursive', '-r', action='store_true', help='Рекурсивный поиск файлов в папках')
    parser.add_argument('--no-whisper', action='store_true', help='Отключить выделение и распознавание вокала')
    parser.add_argument('--whisper-model', default='openai/whisper-large-v3-turbo', help='Модель Whisper')
    parser.add_argument('--theme', '-t', default='', help='Тема текста (перевод, фанфик, оригинал)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный лог')
    parser.add_argument('--quiet', '-q', action='store_true', help='Только ошибки')

    args = parser.parse_args()
    if not args.files:
        parser.print_help()
        sys.exit(0)
    return args


EARLY_ARGS = parse_args_early()

# ==================== ТЯЖЕЛЫЕ ИМПОРТЫ ====================
import json
import logging
import re
import os
import gc
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Optional

import numpy as np
import librosa
from scipy.signal import medfilt
from scipy.fft import fft, ifft, next_fast_len
from sklearn.cluster import AgglomerativeClustering
import essentia
import essentia.standard as es

ESSENTIA_AVAILABLE = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Transformers not installed.")

import soundfile as sf
from tqdm import tqdm
import torch
from demucs import pretrained
from demucs.apply import apply_model

DEMUCS_AVAILABLE = True

__version__ = "6.6.1"


def clear_memory():
    gc.collect()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_logging(verbose: bool = False, quiet: bool = False):
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logger.setLevel(level)
    logging.getLogger('librosa').setLevel(logging.WARNING)
    if ESSENTIA_AVAILABLE: logging.getLogger('essentia').setLevel(logging.WARNING)
    if WHISPER_AVAILABLE: logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)


# ==================== КОНСТАНТЫ ====================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_CLASS_MAP = {name: i for i, name in enumerate(NOTE_NAMES)}
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.0, 2.0, 4.0, 4.0, 2.0, 5.0, 2.0, 3.0, 2.0, 3.0])
TEMPERLEY_MINOR = np.array([5.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 5.0, 3.0, 2.0, 4.0, 2.0])
ALBRECHT_MAJOR = np.array([0.238, 0.006, 0.011, 0.006, 0.074, 0.014, 0.006, 0.241, 0.006, 0.039, 0.006, 0.014])
ALBRECHT_MINOR = np.array([0.220, 0.006, 0.011, 0.074, 0.014, 0.029, 0.006, 0.226, 0.057, 0.014, 0.023, 0.006])
BELLMAN_MAJOR = np.array([0.169, 0.003, 0.041, 0.003, 0.117, 0.013, 0.003, 0.212, 0.007, 0.054, 0.003, 0.027])
BELLMAN_MINOR = np.array([0.181, 0.003, 0.048, 0.074, 0.013, 0.034, 0.003, 0.214, 0.074, 0.013, 0.034, 0.013])
METHOD_WEIGHTS = {
    'Krumhansl-Schmuckler': 0.8, 'Temperley': 1.0, 'Albrecht-Shanahan': 1.3,
    'Bellman': 1.1, 'Bass Analysis': 1.5, 'Track Boundaries': 0.9,
    'Chord Voting': 1.3, 'Circle of Fifths': 0.8, 'Spectral Analysis': 0.7,
    'Essentia': 1.5,
}
AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff'}


# ==================== ТИПИЗАЦИЯ ====================
class MethodResult(TypedDict, total=False):
    method: str;
    pitch_class: int;
    key: str;
    mode: str;
    score: float;
    confidence: float;
    weight: float;
    error: str


@dataclass
class AnalysisConfig:
    sample_rate: int = 44100;
    trim_top_db: int = 20;
    use_hpss: bool = True;
    normalize_audio: bool = True
    bpm_min: int = 40;
    bpm_max: int = 220;
    use_whisper: bool = True
    whisper_model: str = "openai/whisper-large-v3-turbo";
    whisper_cache_dir: Optional[str] = None
    analyze_sections: bool = True;
    analyze_rhythm: bool = True;
    analyze_vocal: bool = True
    analyze_chords: bool = True;
    analyze_dynamics: bool = True;
    analyze_timbre: bool = True
    analyze_genre: bool = True;
    analyze_contour: bool = True;
    analyze_texture: bool = True
    use_neural_api: bool = False;
    neural_api_url: str = "http://localhost:8000/predict"
    vocal_backend: str = "demucs";
    segment_duration_sec: float = 8.0;
    min_section_duration: float = 10.0
    vocal_min_frames: float = 0.25;
    vocal_gap_frames: float = 0.3;
    use_phonetic: bool = True
    use_stress: bool = True;
    max_autocorr_duration: float = 60.0;
    chroma_reduction: Optional[int] = None
    demucs_chunk_sec: float = 7.0;
    theme_prompt: str = ""


@dataclass
class AudioFeatures:
    y: np.ndarray;
    y_perc: np.ndarray;
    y_harm: np.ndarray;
    sr: int;
    chroma: np.ndarray;
    bass_chroma: np.ndarray
    tempo: float;
    beat_frames: np.ndarray;
    onset_env: np.ndarray;
    duration: float
    spectral_centroid: np.ndarray | None = None;
    spectral_contrast: np.ndarray | None = None
    spectral_rolloff: np.ndarray | None = None;
    zero_crossing_rate: np.ndarray | None = None;
    mfccs: np.ndarray | None = None


@dataclass
class ExtendedResult:
    file: str;
    key: str;
    mode: str;
    confidence: float;
    confidence_level: str;
    votes: int;
    total_methods: int
    bpm: dict | None = None;
    duration_seconds: float = 0.0;
    sample_rate: int = 22050
    structure: dict | None = None;
    rhythm: dict | None = None;
    vocal: dict | None = None
    chords: dict | None = None;
    dynamics: dict | None = None;
    timbre: dict | None = None
    genre: dict | None = None;
    neural_analysis: dict | None = None;
    contour: dict | None = None;
    texture: dict | None = None
    theme_hints: str = "";
    theme_prompt: str = "";
    all_results: list = field(default_factory=list);
    voting: dict = field(default_factory=dict);
    error: str | None = None

    def to_dict(self) -> dict: return {k: v for k, v in self.__dict__.items()}


# ==================== УТИЛИТЫ ====================
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


class Colorizer:
    _codes = {'RED': '\033[91m', 'GREEN': '\033[92m', 'YELLOW': '\033[93m', 'BLUE': '\033[94m', 'MAGENTA': '\033[95m',
              'CYAN': '\033[96m', 'WHITE': '\033[97m', 'BOLD': '\033[1m', 'UNDERLINE': '\033[4m', 'RESET': '\033[0m'}

    def __init__(self, enabled: bool = True): self.enabled = enabled and sys.stdout.isatty()

    def wrap(self, text: str, *styles: str) -> str:
        if not self.enabled: return str(text)
        return f"{''.join(self._codes.get(s, '') for s in styles)}{text}{self._codes['RESET']}"


def normalize_key(key_name: str) -> str:
    try:
        return NOTE_NAMES[librosa.note_to_midi(key_name) % 12]
    except Exception:
        mapping = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B', 'Fb': 'E', 'B#': 'C',
                   'E#': 'F'}
        return mapping.get(key_name, key_name)


def normalize_audio(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(y))
    return y if peak == 0 else y * (10 ** (target_db / 20.0) / peak)


def get_audio_files(paths: list[str], recursive: bool = False) -> list[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            files.append(path)
        elif path.is_dir():
            pattern = path.rglob('*') if recursive else path.glob('*')
            files.extend([f for f in pattern if f.is_file() and f.suffix.lower() in AUDIO_EXTS])
    return sorted(list(set(files)))


def soundex(word: str) -> str:
    word = word.upper();
    first = word[0]
    mapping = {'B': '1', 'F': '1', 'P': '1', 'V': '1', 'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2',
               'X': '2', 'Z': '2', 'D': '3', 'T': '3', 'L': '4', 'M': '5', 'N': '5', 'R': '6'}
    code = [first] + [mapping.get(ch, '0') for ch in word[1:]]
    result = []
    for c in code:
        if not result or result[-1] != c: result.append(c)
    return ''.join([result[0]] + [c for c in result[1:] if c != '0'])[:4].ljust(4, '0')


def phonetic_similarity(s1: str, s2: str) -> float: return 1.0 if s1 and s2 and soundex(s1) == soundex(s2) else 0.0


def get_syllables(text: str, lang: str) -> int:
    if not text: return 0
    cyrillic_langs = ['ru', 'uk', 'be', 'bg', 'mn', 'kk', 'ky', 'tg']
    vowels_cyr = 'аеёиоуыэюяАЕЁИОУЫЭЮЯөӨүҮ'
    vowels_lat = 'aeiouyAEIOUY'
    is_cyrillic = lang in cyrillic_langs or any(c in text for c in 'өӨүҮ')
    return len(re.findall(f'[{vowels_cyr}]', text)) if is_cyrillic else len(re.findall(f'[{vowels_lat}]+', text))


def get_rhyme_ending(text: str, lang: str) -> str:
    if not text: return ""
    text = text.strip().lower()
    vowels = 'аеёиоуыэюяөү' if lang in ['ru', 'uk', 'be', 'bg', 'mn'] else 'aeiouy'
    matches = list(re.finditer(f'[{vowels}]', text))
    return text[-3:] if len(text) >= 3 else text if len(matches) < 2 else text[matches[-2].start():]


def rhyme_similarity(ending1: str, ending2: str, use_phonetic: bool = True) -> float:
    if not ending1 or not ending2: return 0.0
    if use_phonetic: return phonetic_similarity(ending1, ending2)
    matches = 0
    for c1, c2 in zip(reversed(ending1), reversed(ending2)):
        if c1 == c2:
            matches += 1
        else:
            break
    return matches / max(len(ending1), len(ending2), 1)


# ==================== КЛАССЫ ТОНАЛЬНОСТИ ====================
class KeyMethod:
    name: str = "BaseMethod";
    weight: float = 1.0

    def __init__(self, features: AudioFeatures):
        self.features = features;
        self._chroma_avg = None

    @property
    def chroma_avg(self) -> np.ndarray:
        if self._chroma_avg is None:
            self._chroma_avg = np.median(self.features.chroma, axis=1) - np.median(
                np.median(self.features.chroma, axis=1))
            norm = np.linalg.norm(self._chroma_avg)
            if norm > 0: self._chroma_avg = self._chroma_avg / norm
        return self._chroma_avg

    def detect(self) -> MethodResult:
        raise NotImplementedError

    def result(self, key_pc, mode, score, confidence):
        return {'method': self.name, 'pitch_class': int(key_pc), 'key': NOTE_NAMES[key_pc], 'mode': mode,
                'score': float(score), 'confidence': float(np.clip(confidence, 0.0, 1.0)), 'weight': self.weight}


class ProfileMethod(KeyMethod):
    _maj_matrices, _min_matrices = {}, {}

    def __init__(self, major_profile, minor_profile, method_name, features):
        super().__init__(features)
        self.name, self.weight = method_name, METHOD_WEIGHTS.get(method_name, 1.0)
        if method_name not in self._maj_matrices:
            maj_norm = (major_profile - np.mean(major_profile));
            maj_norm /= np.linalg.norm(maj_norm)
            min_norm = (minor_profile - np.mean(minor_profile));
            min_norm /= np.linalg.norm(min_norm)
            self._maj_matrices[method_name] = np.array([np.roll(maj_norm, i) for i in range(12)])
            self._min_matrices[method_name] = np.array([np.roll(min_norm, i) for i in range(12)])
        self.maj_matrix, self.min_matrix = self._maj_matrices[method_name], self._min_matrices[method_name]

    def detect(self):
        avg = self.chroma_avg
        maj_corrs, min_corrs = np.dot(self.maj_matrix, avg), np.dot(self.min_matrix, avg)
        bmi, bni = int(np.argmax(maj_corrs)), int(np.argmax(min_corrs))
        if maj_corrs[bmi] > min_corrs[bni]: return self.result(bmi, 'Major', maj_corrs[bmi], (maj_corrs[bmi] + 1) / 2)
        return self.result(bni, 'Minor', min_corrs[bni], (min_corrs[bni] + 1) / 2)


class BassMethod(KeyMethod):
    name, weight = "Bass Analysis", METHOD_WEIGHTS.get("Bass Analysis", 1.0)

    def detect(self):
        try:
            be = np.mean(self.features.bass_chroma, axis=1);
            ti = int(np.argmax(be))
            sc = be[ti] + 0.5 * be[(ti + 7) % 12]
            m = 'Major' if be[(ti + 4) % 12] > be[(ti + 3) % 12] else 'Minor'
            return self.result(ti, m, sc, min(1.0, (sc / (np.max(be) + 1e-10)) * 1.2))
        except Exception as e:
            return {'method': self.name, 'error': str(e)}


class BoundariesMethod(KeyMethod):
    name, weight = "Track Boundaries", METHOD_WEIGHTS.get("Track Boundaries", 1.0)

    def detect(self):
        w = min(10, self.features.duration * 0.1)
        if w < 0.5: return {'method': self.name, 'error': 'Track too short'}
        sf = librosa.time_to_frames(w, sr=self.features.sr)
        if sf * 2 >= self.features.chroma.shape[1]: sf = self.features.chroma.shape[1] // 4
        bc = (np.mean(self.features.chroma[:, :sf], axis=1) + np.mean(self.features.chroma[:, -sf:], axis=1)) / 2
        bc -= np.mean(bc);
        ti = int(np.argmax(bc))
        me, mn = bc[(ti + 4) % 12], bc[(ti + 3) % 12]
        return self.result(ti, 'Major' if me > mn else 'Minor', bc[ti],
                           min(1.0, abs(me - mn) / (abs(me) + abs(mn) + 1e-10) * 2))


class ChordVotingMethod(KeyMethod):
    name, weight = "Chord Voting", METHOD_WEIGHTS.get("Chord Voting", 1.0)

    def detect(self):
        if len(self.features.beat_frames) < 8: return {'method': self.name, 'error': 'Not enough beats'}
        step = 4 if len(self.features.beat_frames) > 16 else 2
        tv, mv = np.zeros(12, dtype=int), {'Major': 0, 'Minor': 0}
        for i in range(0, len(self.features.beat_frames) - step, step):
            s, e = self.features.beat_frames[i], min(self.features.beat_frames[i + step], self.features.chroma.shape[1])
            if s >= e: continue
            ac = np.mean(self.features.chroma[:, s:e], axis=1);
            pi = int(np.argmax(ac))
            tv[pi] += 1
            tm, tmn, f = ac[(pi + 4) % 12], ac[(pi + 3) % 12], ac[(pi + 7) % 12]
            if tm > tmn and f > 0.3 * ac[pi]:
                mv['Major'] += 1
            elif tmn > tm and f > 0.3 * ac[pi]:
                mv['Minor'] += 1
            else:
                mv['Major' if tm > tmn else 'Minor'] += 1
        if np.sum(tv) == 0: return {'method': self.name, 'error': 'No votes'}
        mc = int(np.argmax(tv));
        m = 'Major' if mv['Major'] >= mv['Minor'] else 'Minor'
        return self.result(mc, m, tv[mc] / np.sum(tv), mv[m] / (mv['Major'] + mv['Minor'] + 1e-10))


class CircleOfFifthsMethod(KeyMethod):
    name, weight = "Circle of Fifths", METHOD_WEIGHTS.get("Circle of Fifths", 1.0)

    def detect(self):
        avg, idx = self.chroma_avg, np.arange(12)
        ms = avg[idx] * 2.0 + avg[(idx + 7) % 12] * 1.5 + avg[(idx + 5) % 12]
        mns = avg[idx] * 2.0 + avg[(idx + 7) % 12] * 1.5 + avg[(idx + 3) % 12]
        bmi, bmi2 = int(np.argmax(ms)), int(np.argmax(mns))
        if ms[bmi] > mns[bmi2]: return self.result(bmi, 'Major', ms[bmi], min(1.0, ms[bmi] / (np.max(avg) * 3 + 1e-10)))
        return self.result(bmi2, 'Minor', mns[bmi2], min(1.0, mns[bmi2] / (np.max(avg) * 3 + 1e-10)))


class SpectralMethod(KeyMethod):
    name, weight = "Spectral Analysis", METHOD_WEIGHTS.get("Spectral Analysis", 1.0)

    def detect(self):
        avg = self.chroma_avg;
        ti = int(np.argmax(avg));
        td = avg[(ti + 4) % 12] - avg[(ti + 3) % 12]
        mc = np.mean(self.features.spectral_centroid) if self.features.spectral_centroid is not None else 0
        mct = np.mean(self.features.spectral_contrast[1:]) if self.features.spectral_contrast is not None and \
                                                              self.features.spectral_contrast.shape[0] > 1 else 0
        m = ('Major' if td > 0 else 'Minor') if mc > 1500 and mct > 0.5 else ('Minor' if td < 0 else 'Major')
        return self.result(ti, m, avg[ti], min(1.0, abs(td) * 3 + 0.2))


class EssentiaMethod(KeyMethod):
    name, weight = "Essentia", METHOD_WEIGHTS.get("Essentia", 1.0)

    def detect(self):
        if not ESSENTIA_AVAILABLE: return {'method': self.name, 'error': 'Essentia not installed'}
        try:
            audio = self.features.y.astype(np.float32);
            br, bs = None, 0.0
            for p in ['temperley', 'krumhansl', 'edma']:
                try:
                    k, s, st = es.KeyExtractor(profileType=p)(audio)
                    if st > bs and k and s: bs, br = st, (k, s, st)
                except:
                    continue
            if not br: return {'method': self.name, 'error': 'Failed'}
            k, s, st = br;
            kn = normalize_key(k)
            if kn not in PITCH_CLASS_MAP: return {'method': self.name, 'error': f'Unknown key: {k}'}
            return self.result(PITCH_CLASS_MAP[kn], s.capitalize(), st, st)
        except Exception as e:
            return {'method': self.name, 'error': str(e)}


# ==================== BPM ====================
class BPMDetector:
    @staticmethod
    def _autocorrelate_bpm_fft(signal, sr, min_bpm=40, max_bpm=220, max_duration_sec=60, hop_length=512):
        max_frames = int(max_duration_sec * sr / hop_length)
        if len(signal) > max_frames: signal = signal[:max_frames]
        min_frames = int(2 * sr / hop_length)
        if len(signal) < min_frames: return None
        signal = signal - np.mean(signal);
        n = len(signal);
        fn = next_fast_len(2 * n)
        fa = ifft(fft(signal, n=fn) * np.conj(fft(signal, n=fn)));
        ac = np.real(fa[:n])
        min_lag_frames = int(60 * sr / max_bpm / hop_length)
        max_lag_frames = int(60 * sr / min_bpm / hop_length)
        if max_lag_frames >= len(ac) or min_lag_frames >= max_lag_frames: return None
        search_range = ac[min_lag_frames:max_lag_frames]
        if len(search_range) == 0: return None
        best_lag_frames = np.argmax(search_range) + min_lag_frames
        bpm = 60 * sr / (best_lag_frames * hop_length)
        return float(bpm) if min_bpm < bpm < max_bpm else None

    @staticmethod
    def detect(features, config):
        results = {};
        y, sr = features.y_perc, features.sr
        if features.tempo: results['librosa'] = float(features.tempo)
        try:
            for sig in [np.mean(np.abs(librosa.stft(y)), axis=0),
                        np.maximum(np.diff(np.mean(np.abs(librosa.stft(y)), axis=0)), 0), features.onset_env]:
                bpm = BPMDetector._autocorrelate_bpm_fft(sig, sr, config.bpm_min, config.bpm_max,
                                                         config.max_autocorr_duration, hop_length=512)
                if bpm: results['auto'] = bpm
        except:
            pass
        if results:
            v = list(results.values());
            m = np.median(v)
            f = {k: val for k, val in results.items() if abs(val - m) / (m + 1e-10) < 0.15}
            return f if f else results
        return {}


# ==================== АНАЛИЗАТОРЫ ====================
class SectionAnalyzer:
    def __init__(self, features: AudioFeatures, config: AnalysisConfig):
        self.features, self.config = features, config

    def analyze(self) -> dict:
        try:
            c, sr, hl = self.features.chroma, self.features.sr, 512

            rms_frames = librosa.feature.rms(y=self.features.y, frame_length=2048, hop_length=hl)[0]
            min_len = min(c.shape[1], len(rms_frames))
            if min_len < 20: return {'sections': [], 'structure_map': 'Too short', 'chorus_count': 0}

            c = c[:, :min_len]
            rms_frames = rms_frames[:min_len]

            beat_times = librosa.frames_to_time(self.features.beat_frames, sr=sr)
            if len(beat_times) >= 4:
                bar_duration = np.median(np.diff(beat_times)) * 4
            else:
                bar_duration = 4 * (60.0 / (self.features.tempo if self.features.tempo > 0 else 120))

            segment_duration = bar_duration * 4
            sf = int(segment_duration * sr / hl)
            if sf < 10: sf = int(8.0 * sr / hl)

            segs, times = [], []
            for i in range(0, min_len - sf, sf):
                seg_chroma = np.mean(c[:, i:i + sf], axis=1)
                seg_rms = np.mean(rms_frames[i:i + sf])
                seg_chroma = seg_chroma / (np.linalg.norm(seg_chroma) + 1e-10)
                segs.append(np.concatenate([seg_chroma, [seg_rms]]))
                start_t = librosa.frames_to_time(i, sr=sr, hop_length=hl)
                end_t = librosa.frames_to_time(i + sf, sr=sr, hop_length=hl)
                times.append((start_t, end_t))

            if len(segs) < 3: return {'sections': [], 'structure_map': 'Too short', 'chorus_count': 0}

            X = np.array(segs)
            max_rms = np.max(X[:, 12])
            if max_rms > 0: X[:, 12] = X[:, 12] / max_rms

            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.2, metric='euclidean',
                                                 linkage='ward')
            labels = clustering.fit_predict(X)

            cluster_stats = {}
            for c_id in set(labels):
                indices = [i for i, l in enumerate(labels) if l == c_id]
                cluster_stats[c_id] = {
                    'count': len(indices),
                    'energy': np.mean([segs[i][12] for i in indices]),
                    'positions': indices
                }

            sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['energy'], reverse=True)
            median_energy = np.median([s['energy'] for s in cluster_stats.values()])

            label_map = {}
            chorus_assigned = False
            verse_assigned = False

            for c_id, stats in sorted_clusters:
                if stats['positions'][0] == 0 and stats['count'] == 1:
                    label_map[c_id] = "Intro"
                elif stats['positions'][-1] == len(labels) - 1 and stats['count'] == 1:
                    label_map[c_id] = "Outro"
                elif stats['count'] >= 2 and not chorus_assigned and stats['energy'] > median_energy:
                    label_map[c_id] = "Chorus"
                    chorus_assigned = True
                elif stats['count'] >= 2 and not verse_assigned:
                    label_map[c_id] = "Verse"
                    verse_assigned = True
                elif stats['count'] == 1:
                    label_map[c_id] = "Bridge"
                else:
                    label_map[c_id] = "Verse"

            raw_sections = []
            for i, (label_id, (start_t, end_t)) in enumerate(zip(labels, times)):
                raw_sections.append({
                    'label': label_map.get(label_id, "Unknown"),
                    'start_time': round(start_t, 2),
                    'end_time': round(end_t, 2),
                    'duration': round(end_t - start_t, 2),
                    'cluster_id': int(label_id),
                    'energy': round(float(segs[i][12]), 3)
                })

            merged = []
            for s in raw_sections:
                if not merged:
                    merged.append(s.copy())
                else:
                    la = merged[-1]
                    if la['label'] == s['label']:
                        la['end_time'] = s['end_time']
                        la['duration'] = round(s['end_time'] - la['start_time'], 2)
                    else:
                        merged.append(s.copy())

            # === ПРАВКА v6.6.1 #1: Разбивка секций >32с пополам ===
            final_sections = []
            for s in merged:
                if s['duration'] > 32.0:
                    sp = s['start_time'] + s['duration'] / 2
                    final_sections.append({
                        'label': s['label'], 'start_time': s['start_time'], 'end_time': round(sp, 2),
                        'duration': round(sp - s['start_time'], 2), 'cluster_id': s['cluster_id'], 'energy': s['energy']
                    })
                    final_sections.append({
                        'label': s['label'], 'start_time': round(sp, 2), 'end_time': s['end_time'],
                        'duration': round(s['end_time'] - sp, 2), 'cluster_id': s['cluster_id'], 'energy': s['energy']
                    })
                else:
                    final_sections.append(s)

            be = self.features.tempo if self.features.tempo > 0 else 120;
            bs = 60.0 / be
            for s in final_sections:
                b = s['duration'] / bs
                s['estimated_bars'], s['estimated_beats'] = round(b / 4.0, 1), round(b, 1)

            return {
                'sections': final_sections,
                'structure_map': " → ".join([s['label'] for s in final_sections]),
                'chorus_count': sum(1 for s in final_sections if s['label'] == "Chorus"),
                'total_sections': len(final_sections),
                'unique_clusters': len(set(labels))
            }
        except Exception as e:
            return {'error': str(e), 'sections': []}


class RhythmAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            if self.features.tempo <= 0 or len(self.features.beat_frames) < 4: return {
                'error': 'Cannot determine rhythm'}
            bt = librosa.frames_to_time(self.features.beat_frames, sr=self.features.sr);
            iv = np.diff(bt)
            mi, iv2 = np.median(iv), np.var(iv) / (np.median(iv) ** 2)
            if len(self.features.beat_frames) > 4:
                be = [self.features.onset_env[min(b, len(self.features.onset_env) - 1)] for b in
                      self.features.beat_frames]
                sc = {}
                for gs in [2, 3, 4]:
                    g = [be[i:i + gs] for i in range(0, len(be) - len(be) % gs, gs)]
                    if g: sc[gs] = np.mean([x[0] for x in g]) / (
                            np.mean([np.mean(x[1:]) for x in g if len(x) > 1]) + 1e-10)
                b = max(sc, key=sc.get)
                m = f"{b}/4" if b in [2, 4] else "3/4" if b == 3 else "4/4" if sc[b] <= 1.2 else "Free"
            else:
                m = "4/4" if iv2 < 0.01 else "Free"
            if m == "2/4" and self.features.tempo < 160: m = "4/4"
            return {'tempo_bpm': round(float(self.features.tempo), 1), 'meter': m,
                    'beat_count': len(self.features.beat_frames), 'regularity': round(float(max(0, 1 - iv2 * 10)), 3),
                    'syncopation': round(float(max(0, 1 - (np.mean(
                        [self.features.onset_env[min(b, len(self.features.onset_env) - 1)] for b in
                         self.features.beat_frames]) / (np.mean(self.features.onset_env) + 1e-10)))), 3)}
        except Exception as e:
            return {'error': str(e)}


class ChordAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            maj_prof = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
            min_prof = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
            chords = []
            if len(self.features.beat_frames) < 8: return {'chord_progression': [], 'total_chords': 0}
            step = 4
            for i in range(0, len(self.features.beat_frames) - step, step):
                s = self.features.beat_frames[i];
                e = self.features.beat_frames[min(i + step, len(self.features.beat_frames) - 1)]
                if s >= e or e > self.features.chroma.shape[1]: continue
                c_vec = np.mean(self.features.chroma[:, s:e], axis=1)
                c_vec = c_vec / (np.max(c_vec) + 1e-10)
                scores = []
                for root in range(12):
                    scores.append((NOTE_NAMES[root], 'Major', np.dot(np.roll(maj_prof, root), c_vec)))
                    scores.append((NOTE_NAMES[root], 'Minor', np.dot(np.roll(min_prof, root), c_vec)))
                best = max(scores, key=lambda x: x[2])
                chords.append({'chord': f"{best[0]}{'m' if best[1] == 'Minor' else ''}",
                               'start_time': round(float(librosa.frames_to_time(s, sr=self.features.sr)), 2),
                               'end_time': round(float(librosa.frames_to_time(e, sr=self.features.sr)), 2),
                               'confidence': round(float(best[2]), 2)})
            return {'chord_progression': chords, 'total_chords': len(chords)}
        except Exception as e:
            return {'error': str(e)}


class DynamicsAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            rms = librosa.feature.rms(y=self.features.y, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
            fps = self.features.sr / 512.0
            rms_sec = [np.mean(rms[max(0, int(i * fps - fps // 2)):int(i * fps + fps // 2)]) for i in
                       range(int(self.features.duration))]
            climax_time = int(np.argmax(rms_sec)) if rms_sec else 0
            drops = []
            rms_sec_arr = np.array(rms_sec)
            for i in range(1, len(rms_sec_arr)):
                if rms_sec_arr[i - 1] > 0.01 and rms_sec_arr[i] < rms_sec_arr[i - 1] * 0.5: drops.append(i)
            return {'average_loudness_db': round(float(np.mean(rms_db)), 1),
                    'dynamic_range_db': round(dynamic_range, 1), 'climax_time_sec': climax_time,
                    'drops_count': len(drops), 'drops_times_sec': drops[:5]}
        except Exception as e:
            return {'error': str(e)}


class ContourAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            if len(self.features.beat_frames) < 8: return {'contour': 'unknown'}
            pitches = []
            step = 4
            for i in range(0, len(self.features.beat_frames) - step, step):
                s = self.features.beat_frames[i];
                e = self.features.beat_frames[min(i + step, len(self.features.beat_frames) - 1)]
                if s >= e or e > self.features.chroma.shape[1]: continue
                c_vec = np.mean(self.features.chroma[:, s:e], axis=1)
                pitches.append(int(np.argmax(c_vec)))
            if len(pitches) < 3: return {'contour': 'unknown'}
            x = np.arange(len(pitches));
            slope = np.polyfit(x, pitches, 1)[0]
            if slope > 0.1:
                contour = "ascending"
            elif slope < -0.1:
                contour = "descending"
            else:
                contour = "arch/static"
            return {'contour': contour, 'slope': round(float(slope), 3)}
        except Exception as e:
            return {'error': str(e)}


class TimbreAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            cent = np.mean(self.features.spectral_centroid) if self.features.spectral_centroid is not None else 0
            if cent < 1000:
                timbre = "dark/deep"
            elif cent < 2500:
                timbre = "warm"
            elif cent < 4500:
                timbre = "bright"
            else:
                timbre = "harsh/metallic"
            return {'spectral_centroid_hz': round(float(cent), 1), 'timbre_class': timbre}
        except Exception as e:
            return {'error': str(e)}


class TextureAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            zcr = np.mean(self.features.zero_crossing_rate) if self.features.zero_crossing_rate is not None else 0
            contrast = np.mean(self.features.spectral_contrast) if self.features.spectral_contrast is not None else 0
            rms = librosa.feature.rms(y=self.features.y, frame_length=2048, hop_length=512)[0]
            rms_variance = np.var(rms)
            if rms_variance > 0.01 or zcr > 0.04:
                texture = "dense/distorted (Heavy Rock/Folk Metal)"
            elif contrast > 25:
                texture = "harmonic/clean (Acoustic/Folk)"
            else:
                texture = "sparse"
            return {'avg_zcr': round(float(zcr), 4), 'texture_class': texture,
                    'rms_variance': round(float(rms_variance), 4)}
        except Exception as e:
            return {'error': str(e)}


class GenreAnalyzer:
    def __init__(self, features: AudioFeatures, vocal_result: dict | None = None):
        self.features = features
        self.vocal_result = vocal_result or {}

    def analyze(self) -> dict:
        try:
            if self.vocal_result.get('throat_singing_likely'):
                genre = "Hunnu Rock / Folk Metal"
            elif self.features.spectral_centroid is not None and np.mean(self.features.spectral_centroid) > 3000:
                genre = "Metal / Hard Rock"
            else:
                genre = "Folk Rock / Alternative"
            return {'genre_class': genre, 'confidence': 'heuristic'}
        except Exception as e:
            return {'error': str(e)}


# ==================== АНАЛИЗАТОР ВОКАЛА ====================
class VocalAnalyzer:
    _whisper_pipe_cache, _demucs_model_cache = {}, None

    def __init__(self, features: AudioFeatures, config: AnalysisConfig):
        self.features, self.config = features, config
        self.use_whisper = config.use_whisper and WHISPER_AVAILABLE
        self.whisper_model_name, self.vocal_backend = config.whisper_model, config.vocal_backend
        self._vocal_audio = None

    def _get_whisper_pipe(self):
        if not WHISPER_AVAILABLE: return None
        cd = self.config.whisper_cache_dir or os.environ.get("WHISPER_CACHE_DIR")
        if self.whisper_model_name not in self._whisper_pipe_cache:
            logger.info(f"🔄 Загрузка Whisper: '{self.whisper_model_name}'")
            dev = "cuda:0" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps",
                                                                              None) is not None and torch.backends.mps.is_available() else "cpu"
            td = torch.float16 if dev == "cuda:0" else torch.float32
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.whisper_model_name, dtype=td, low_cpu_mem_usage=True,
                                                              cache_dir=cd).to(dev)
            proc = AutoProcessor.from_pretrained(self.whisper_model_name, cache_dir=cd)
            self._whisper_pipe_cache[self.whisper_model_name] = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
                chunk_length_s=30,
                batch_size=1,
                dtype=td,
                device=dev,
                ignore_warning=True
            )
            logger.info("✅ Whisper загружен.")
        return self._whisper_pipe_cache[self.whisper_model_name]

    def _get_demucs_model(self):
        if not DEMUCS_AVAILABLE: return None
        if self._demucs_model_cache is None: self._demucs_model_cache = pretrained.get_model('htdemucs').eval()
        return self._demucs_model_cache

    def _detect_vocal_presence(self) -> dict:
        y, sr = self.features.y, self.features.sr

        def count_contiguous_segments(mask):
            if len(mask) == 0 or not np.any(mask): return 0
            padded = np.concatenate(([False], mask, [False]))
            diffs = np.diff(padded.astype(int))
            return int(np.sum(diffs == 1))

        if self.vocal_backend != 'demucs' or not DEMUCS_AVAILABLE:
            y_h, y_p = librosa.effects.hpss(y);
            self._vocal_audio = y_h
            win = int(1.0 * sr)
            h_rms = np.array([np.sqrt(np.mean(y_h[i:i + win] ** 2)) for i in range(0, len(y_h), win)])
            p_rms = np.array([np.sqrt(np.mean(y_p[i:i + win] ** 2)) for i in range(0, len(y_p), win)])
            v_thresh = np.percentile(h_rms[h_rms > 0], 15) if np.any(h_rms > 0) else 0.01
            is_vocal = (h_rms > v_thresh) & (h_rms > p_rms * 0.4)
            is_inst = (p_rms > 0.01) | ((h_rms > v_thresh) & (p_rms > h_rms * 1.2))
            return {'method': 'hpss', 'has_vocals': bool(np.any(is_vocal)),
                    'vocal_rms': float(np.sqrt(np.mean(y_h ** 2))),
                    'vocal_segments_count': count_contiguous_segments(is_vocal),
                    'instrument_segments_count': count_contiguous_segments(is_inst)}
        try:
            model = self._get_demucs_model();
            device = 'cpu';
            model.to(device)
            y_st = np.stack([y, y]) if y.ndim == 1 else y
            cs = int(self.config.demucs_chunk_sec * sr);
            vc = []
            window_sec = 1.0;
            window_samples = int(window_sec * sr)
            vocal_energies = [];
            inst_energies = []
            logger.info(f"🎙️ Разделение вокала (чанки по {self.config.demucs_chunk_sec}с на CPU)...")
            for i in tqdm(range(0, y_st.shape[1], cs), desc="Demucs", leave=False):
                chunk = torch.from_numpy(y_st[:, i:i + cs]).float().unsqueeze(0)
                with torch.no_grad():
                    src = apply_model(model, chunk, device=device, shifts=0, split=False)
                vocal_chunk = src[0, 3].mean(dim=0).numpy();
                inst_chunk = src[0, 0:3].mean(dim=0).mean(dim=0).numpy()
                for w_start in range(0, len(vocal_chunk), window_samples):
                    w_end = min(w_start + window_samples, len(vocal_chunk))
                    vocal_energies.append(np.sqrt(np.mean(vocal_chunk[w_start:w_end] ** 2)) if w_end > w_start else 0)
                    inst_energies.append(np.sqrt(np.mean(inst_chunk[w_start:w_end] ** 2)) if w_end > w_start else 0)
                vc.append(vocal_chunk);
                del chunk, src;
                clear_memory()
            vocal = np.concatenate(vc);
            self._vocal_audio = vocal;
            del vc;
            clear_memory()
            vocal_energies = np.array(vocal_energies);
            inst_energies = np.array(inst_energies)
            v_thresh = np.percentile(vocal_energies[vocal_energies > 0], 15) if np.any(vocal_energies > 0) else 0.005
            i_thresh = np.percentile(inst_energies[inst_energies > 0], 15) if np.any(inst_energies > 0) else 0.005
            is_vocal = (vocal_energies > v_thresh) & (vocal_energies > inst_energies * 0.3)
            is_inst = (inst_energies > i_thresh) & (inst_energies > vocal_energies * 1.2)
            return {'method': 'demucs_cpu_chunked', 'has_vocals': bool(np.any(is_vocal)),
                    'vocal_rms': float(np.sqrt(np.mean(vocal ** 2))),
                    'vocal_segments_count': int(count_contiguous_segments(is_vocal)),
                    'instrument_segments_count': int(count_contiguous_segments(is_inst)),
                    'total_segments_analyzed': len(vocal_energies),
                    'note': 'CPU chunks used. Optimized for folk/throat singing.'}
        except Exception as e:
            logger.error(f"Demucs failed: {e}");
            y_h, y_p = librosa.effects.hpss(y);
            self._vocal_audio = y_h
            return {'method': 'hpss_fallback', 'error': str(e), 'has_vocals': True, 'vocal_segments_count': 1,
                    'instrument_segments_count': 1}

    def _transcribe_vocals(self) -> dict:
        if not self.use_whisper or self._vocal_audio is None: return {'text': '', 'language': None, 'error': 'Disabled'}
        pipe = self._get_whisper_pipe()
        if not pipe: return {'text': '', 'error': 'No pipe'}
        try:
            v16 = librosa.resample(self._vocal_audio, orig_sr=self.features.sr, target_sr=16000)
            logger.info("📝 Транскрибация...")
            res_auto = pipe(v16, return_timestamps=True, generate_kwargs={"task": "transcribe"})
            lang_auto = res_auto.get("language", "unknown")
            txt_auto = " ".join([c["text"].strip() for c in res_auto.get("chunks", []) if c["text"].strip()])
            words = txt_auto.split()
            is_hallucination = False
            if len(words) > 5 and len(set(words)) / len(words) < 0.3: is_hallucination = True

            if is_hallucination or lang_auto not in ['mn', 'ru', 'en', 'uk', 'be']:
                logger.info("🇲🇳 Обнаружено горловое пение или неясный язык. Форсируем Монгольский (mn)...")
                res_mn = pipe(v16, return_timestamps=True, generate_kwargs={"task": "transcribe", "language": "mn"})
                txt_mn = " ".join([c["text"].strip() for c in res_mn.get("chunks", []) if c["text"].strip()])
                if len(txt_mn.split()) > len(words) or len(txt_mn) > len(txt_auto):
                    res, txt, lang = res_mn, txt_mn, "mn"
                else:
                    res, txt, lang = res_auto, txt_auto, lang_auto
            else:
                res, txt, lang = res_auto, txt_auto, lang_auto

            del v16;
            clear_memory()

            chunks = res.get("chunks", [])

            beat_times = librosa.frames_to_time(self.features.beat_frames, sr=self.features.sr)
            if len(beat_times) >= 4:
                bar_duration = np.median(np.diff(beat_times)) * 4
            else:
                bar_duration = 4 * (60.0 / (self.features.tempo if self.features.tempo > 0 else 120))

            bar_times = np.arange(0, self.features.duration + bar_duration, bar_duration)

            syllable_grid = []
            lines = []
            current_line = {'text': '', 'syllables': 0, 'start_time': 0, 'end_time': 0, 'break_reason': 'start'}
            prev_word_end = 0.0
            line_start_bar = 0

            for c in chunks:
                text = c.get("text", "").strip()
                if not text: continue
                timestamp = c.get("timestamp", (0, 0))
                start = timestamp[0] if timestamp[0] is not None else prev_word_end
                end = timestamp[1] if timestamp[1] is not None else start + 1.0

                start_bar_idx = np.searchsorted(bar_times, start)
                end_bar_idx = np.searchsorted(bar_times, end)

                pause = start - prev_word_end

                if not current_line['text']:
                    current_line['start_time'] = start
                    line_start_bar = start_bar_idx

                bars_in_line = end_bar_idx - line_start_bar
                current_syllables = current_line['syllables'] + get_syllables(text, lang)

                # === ПРАВКА v6.6.1 #4: Порог паузы 1.5с → 0.8с ===
                new_line = False
                break_reason = ""
                if pause > 0.8 and current_line['syllables'] > 0:
                    new_line = True;
                    break_reason = "pause"
                elif current_line['syllables'] >= 6 and bars_in_line >= 2:
                    new_line = True;
                    break_reason = "bars"
                elif bars_in_line >= 4 and current_line['syllables'] > 0:
                    new_line = True;
                    break_reason = "bars"
                elif current_syllables > 16:
                    new_line = True;
                    break_reason = "syllables"

                if new_line and current_line['syllables'] > 0:
                    current_line['end_time'] = prev_word_end
                    current_line['break_reason'] = break_reason
                    lines.append(current_line)
                    current_line = {'text': text, 'syllables': get_syllables(text, lang), 'start_time': start,
                                    'end_time': end, 'break_reason': 'start'}
                    line_start_bar = start_bar_idx
                else:
                    current_line['text'] += (" " if current_line['text'] else "") + text
                    current_line['syllables'] = current_syllables
                    current_line['end_time'] = end

                syllable_grid.append({'text': text, 'syllables': get_syllables(text, lang), 'start': round(start, 2),
                                      'end': round(end, 2)})
                prev_word_end = end

            if current_line['syllables'] > 0:
                current_line['end_time'] = prev_word_end
                current_line['break_reason'] = "end"
                lines.append(current_line)

            meter_info = "Не определена"
            avg_syllables = 0
            rhyme_scheme = []
            rhyme_pattern = "Free"

            if lines:
                syl_counts = [l['syllables'] for l in lines if 3 <= l['syllables'] <= 20]
                if syl_counts:
                    avg_syllables = np.mean(syl_counts)
                    common_meters = [6, 8, 10, 11, 12, 14]
                    best_meter = min(common_meters, key=lambda x: abs(x - avg_syllables))
                    meter_info = f"~{best_meter} слогов (среднее {avg_syllables:.1f})"

                endings = []
                for l in lines:
                    words_list = l['text'].split()
                    if words_list:
                        endings.append(get_rhyme_ending(words_list[-1], lang))
                    else:
                        endings.append("")

                letters = {}
                next_letter = ord('A')
                for end in endings:
                    if not end:
                        rhyme_scheme.append('?')
                        continue
                    found = False
                    for known_end, letter_code in letters.items():
                        if rhyme_similarity(end, known_end, use_phonetic=True) > 0.6:
                            rhyme_scheme.append(chr(letter_code))
                            found = True
                            break
                    if not found:
                        rhyme_scheme.append(chr(next_letter))
                        letters[end] = next_letter
                        next_letter += 1
                        if next_letter > ord('Z'): next_letter = ord('A')

                scheme_str = "".join(rhyme_scheme).replace('?', '')
                if len(scheme_str) >= 4:
                    if scheme_str.startswith("AABB") or "AABB" in scheme_str:
                        rhyme_pattern = "AABB (Парная)"
                    elif scheme_str.startswith("ABAB") or "ABAB" in scheme_str:
                        rhyme_pattern = "ABAB (Перекрестная)"
                    elif scheme_str.startswith("ABBA") or "ABBA" in scheme_str:
                        rhyme_pattern = "ABBA (Опоясывающая)"
                    elif len(set(scheme_str)) == 1:
                        rhyme_pattern = "AAAA (Моно-рифма)"

            # === ПРАВКА v6.6.1 #3: lines_per_verse ===
            lines_per_verse = 4
            if len(lines) >= 8:
                lines_per_verse = 4 if len(lines) % 4 == 0 else (8 if len(lines) % 8 == 0 else 4)

            throat_singing_likely = is_hallucination or (
                    not txt and self._vocal_audio is not None and np.sqrt(np.mean(self._vocal_audio ** 2)) > 0.01)

            return {
                'text': txt,
                'language': lang,
                'word_count': len(txt.split()),
                'syllable_count': get_syllables(txt, lang),
                'chunks_count': len(chunks),
                'is_hallucination': is_hallucination,
                'syllable_grid': syllable_grid,
                'lines': lines,
                'lines_count': len(lines),
                'meter': meter_info,
                'avg_syllables_per_line': round(avg_syllables, 1),
                'rhyme_scheme': rhyme_scheme,
                'rhyme_pattern': rhyme_pattern,
                'lines_per_verse': lines_per_verse,
                'throat_singing_likely': throat_singing_likely,
                'bar_duration_sec': round(float(bar_duration), 2)
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {'text': '', 'error': str(e)}

    def analyze(self) -> dict:
        pr = self._detect_vocal_presence()
        tr = self._transcribe_vocals() if pr.get('has_vocals', False) else {}
        return {**pr, **tr}


# ==================== ОСНОВНОЙ АНАЛИЗАТОР ====================
class MusicAnalyzer:
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()

    def _generate_theme_hints(self, res_dict: dict) -> str:
        hints = []
        user_theme = res_dict.get('theme_prompt', '')
        if user_theme:
            hints.insert(0, f"Запрос: {user_theme}")

        genre = res_dict.get('genre', {}).get('genre_class', '')
        if 'Hunnu' in genre or 'Folk' in genre:
            hints.extend(["эпическая", "историческая", "кочевники", "степь", "призыв к битве"])
        if res_dict.get('timbre', {}).get('timbre_class') == 'dark/deep':
            hints.append("мрачная, героическая")
        if res_dict.get('texture', {}).get('texture_class', '').startswith('dense'):
            hints.append("агрессивная, стена звука, протест")
        if res_dict.get('vocal', {}).get('throat_singing_likely'):
            hints.extend(["шаманизм", "дух предков", "горловое пение"])
        if res_dict.get('dynamics', {}).get('dynamic_range_db', 0) > 15:
            hints.append("контрастная, нарастающая ярость")

        return ", ".join(hints) if hints else "нейтральная, повествовательная"

    def _extract_features(self, file_path: str) -> AudioFeatures:
        logger.info(f"📂 Загрузка: {file_path}")
        y, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)
        if self.config.normalize_audio: y = normalize_audio(y)
        y_h, y_p = librosa.effects.hpss(y)
        c = librosa.feature.chroma_cqt(y=y_p, sr=sr, bins_per_octave=36)
        bc = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36, fmin=librosa.note_to_hz('C1'), n_octaves=2)
        t, bf = librosa.beat.beat_track(y=y_p, sr=sr);
        tempo = float(np.atleast_1d(t).mean())
        oe = librosa.onset.onset_strength(y=y_p, sr=sr)
        return AudioFeatures(y=y, y_perc=y_p, y_harm=y_h, sr=sr, chroma=c, bass_chroma=bc, tempo=tempo, beat_frames=bf,
                             onset_env=oe, duration=librosa.get_duration(y=y, sr=sr),
                             spectral_centroid=librosa.feature.spectral_centroid(y=y, sr=sr)[0],
                             spectral_contrast=librosa.feature.spectral_contrast(y=y, sr=sr),
                             spectral_rolloff=librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
                             zero_crossing_rate=librosa.feature.zero_crossing_rate(y)[0],
                             mfccs=librosa.feature.mfcc(y=y, sr=sr))

    def analyze_file(self, file_path: str) -> ExtendedResult:
        try:
            f = self._extract_features(file_path)
            methods = [ProfileMethod(KRUMHANSL_MAJOR, KRUMHANSL_MINOR, "Krumhansl-Schmuckler", f),
                       ProfileMethod(TEMPERLEY_MAJOR, TEMPERLEY_MINOR, "Temperley", f),
                       ProfileMethod(ALBRECHT_MAJOR, ALBRECHT_MINOR, "Albrecht-Shanahan", f),
                       ProfileMethod(BELLMAN_MAJOR, BELLMAN_MINOR, "Bellman", f),
                       BassMethod(f), BoundariesMethod(f), ChordVotingMethod(f), CircleOfFifthsMethod(f),
                       SpectralMethod(f), EssentiaMethod(f)]
            res = [m.detect() for m in methods];
            vr = [r for r in res if 'error' not in r]
            if not vr: return ExtendedResult(file=file_path, key="Unknown", mode="Unknown", confidence=0,
                                             confidence_level="None", votes=0, total_methods=len(res),
                                             error="All key methods failed")
            wv = defaultdict(float)
            for r in vr: wv[(r['key'], r['mode'])] += r['confidence'] * r['weight']
            bk, bm = max(wv, key=wv.get);
            tw = sum(wv.values());
            cf = wv[(bk, bm)] / tw if tw > 0 else 0
            cl = "Very High" if cf > 0.8 else "High" if cf > 0.6 else "Medium" if cf > 0.4 else "Low"
            votes = sum(1 for r in vr if r['key'] == bk and r['mode'] == bm)
            er = ExtendedResult(file=file_path, key=bk, mode=bm, confidence=round(cf, 3), confidence_level=cl,
                                votes=votes, total_methods=len(vr),
                                bpm=BPMDetector.detect(f, self.config), duration_seconds=f.duration, sample_rate=f.sr,
                                all_results=res, voting={f"{k[0]} {k[1]}": v for k, v in wv.items()},
                                theme_prompt=self.config.theme_prompt)

            if self.config.analyze_sections: er.structure = SectionAnalyzer(f, self.config).analyze()
            if self.config.analyze_rhythm: er.rhythm = RhythmAnalyzer(f).analyze()
            if self.config.analyze_vocal: er.vocal = VocalAnalyzer(f, self.config).analyze()
            if self.config.analyze_chords: er.chords = ChordAnalyzer(f).analyze()
            if self.config.analyze_dynamics: er.dynamics = DynamicsAnalyzer(f).analyze()
            if self.config.analyze_contour: er.contour = ContourAnalyzer(f).analyze()
            if self.config.analyze_timbre: er.timbre = TimbreAnalyzer(f).analyze()
            if self.config.analyze_texture: er.texture = TextureAnalyzer(f).analyze()
            if self.config.analyze_genre: er.genre = GenreAnalyzer(f, er.vocal).analyze()

            er.theme_hints = self._generate_theme_hints(er.to_dict())

            del f;
            clear_memory()
            return er
        except Exception as e:
            logger.exception("Critical analysis error")
            return ExtendedResult(file=file_path, key="Error", mode="Error", confidence=0, confidence_level="None",
                                  votes=0, total_methods=0, error=str(e))


# ==================== CLI ====================
def process_file(filepath: Path, config: AnalysisConfig, save: bool, output_dir: Path | None = None) -> ExtendedResult:
    analyzer = MusicAnalyzer(config)
    result = analyzer.analyze_file(str(filepath))
    if save and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{filepath.stem}_analysis.json"
        with open(out_file, 'w', encoding='utf-8') as file:
            json.dump(result.to_dict(), file, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    return result


def format_result_output(result: ExtendedResult, colorizer: Colorizer) -> str:
    lines = [
        colorizer.wrap(f"\n🎵 Файл: {result.file}", "BOLD", "WHITE"),
        colorizer.wrap(
            f"Тональность: {result.key} {result.mode} ({result.confidence * 100:.1f}% - {result.confidence_level})",
            "GREEN")
    ]
    if result.theme_prompt: lines.append(colorizer.wrap(f"🎯 Ваш запрос: {result.theme_prompt}", "MAGENTA"))
    lines.append(colorizer.wrap(f"💡 Вайб/Тема: {result.theme_hints}", "CYAN"))

    if result.error: lines.append(colorizer.wrap(f"⚠️ Ошибка: {result.error}", "RED"))
    if result.bpm: lines.append(f"⏱ BPM: {result.bpm}")
    if result.rhythm: lines.append(
        f"🥁 Ритм: {result.rhythm.get('meter', '?')} | Syncopation: {result.rhythm.get('syncopation', 0)}")
    if result.genre: lines.append(f"🎸 Жанр: {result.genre.get('genre_class', '?')}")

    if result.structure and 'structure_map' in result.structure:
        lines.append(f"🏗 Структура: {result.structure['structure_map']}")
        lines.append(f"🧩 Уникальных кластеров: {result.structure.get('unique_clusters', '?')}")

        # === БОНУС v6.6.1: Детальный вывод секций ===
        for s in result.structure.get('sections', []):
            lines.append(
                f"   [{s['label']:8s}] {s['start_time']:6.1f}–{s['end_time']:6.1f}s ({s['duration']:5.1f}s, ~{s.get('estimated_bars', '?'):>4} тактов, E={s.get('energy', '?')})")

    if result.chords and 'chord_progression' in result.chords and result.chords['chord_progression']:
        prog = [c['chord'] for c in result.chords['chord_progression'][:8]]
        lines.append(f"🎸 Аккорды: {' -> '.join(prog)}...")
    if result.timbre: lines.append(f"🎨 Тембр: {result.timbre.get('timbre_class', '?')}")
    if result.texture: lines.append(f"🧱 Текстура: {result.texture.get('texture_class', '?')}")
    if result.dynamics: lines.append(
        f"📈 Динамика: {result.dynamics.get('average_loudness_db', 0)} dB | Пик: {result.dynamics.get('climax_time_sec', 0)}с | Дропы: {result.dynamics.get('drops_count', 0)}")

    if result.vocal:
        if result.vocal.get('text'):
            lines.append(f"🎤 Текст: {result.vocal['text'][:100]}...")
            lines.append(
                f"📏 Метрика: {result.vocal.get('meter', '?')} | Такт: {result.vocal.get('bar_duration_sec', '?')}с | Строк: {result.vocal.get('lines_count', 0)}")
            lines.append(
                f"🎼 Рифмовка: {result.vocal.get('rhyme_pattern', '?')} ({''.join(result.vocal.get('rhyme_scheme', []))})")
            lines.append(f"📐 Строк в куплете: ~{result.vocal.get('lines_per_verse', '?')}")
        elif result.vocal.get('throat_singing_likely'):
            lines.append(f"🎤 Вокал: Обнаружено горловое пение (текст не распознан)")
        elif not result.vocal.get('has_vocals'):
            lines.append(f"🎤 Вокал: Не обнаружен")
        elif result.vocal.get('has_vocals') and not result.vocal.get('text'):
            lines.append(f"🎤 Вокал: Обнаружен, но текст не распознан (возможно, скрим)")
        lines.append(
            f"🧩 Сегменты: Вокал ({result.vocal.get('vocal_segments_count', 0)}) | Инструментал ({result.vocal.get('instrument_segments_count', 0)})")

    return "\n".join(lines)


def main(args):
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    colorizer = Colorizer(enabled=not args.no_color)
    import torch;
    import multiprocessing as mp
    torch.set_num_threads(1)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    config = AnalysisConfig(
        use_whisper=not args.no_whisper,
        whisper_model=args.whisper_model,
        theme_prompt=args.theme
    )
    files_to_process = get_audio_files(args.files, recursive=args.recursive)
    if not files_to_process:
        logger.error("Не найдено аудиофайлов.")
        print(colorizer.wrap("Не найдено аудиофайлов.", 'RED'))
        return
    if args.json: logging.disable(logging.CRITICAL)
    for filepath in tqdm(files_to_process, desc=f"Обработка v{__version__}", file=sys.stdout, dynamic_ncols=True,
                         disable=args.json):
        result = process_file(Path(filepath), config, not args.no_save, Path(args.output) if args.output else None)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, cls=NumpyJSONEncoder))
        else:
            tqdm.write(format_result_output(result, colorizer))


if __name__ == "__main__":
    main(EARLY_ARGS)
