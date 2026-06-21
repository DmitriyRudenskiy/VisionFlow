from __future__ import annotations

import json
import sys
import argparse
import logging
import warnings
import tempfile
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict, Any
import numpy as np

# Отключаем специфичные предупреждения до импорта librosa
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== ВЕРСИЯ ====================
__version__ = "3.1.0"

# ==================== ЗАВИСИМОСТИ ====================
try:
    import librosa
except ImportError:
    print("Ошибка: Библиотека librosa не установлена. Установите: pip install librosa")
    sys.exit(1)

try:
    from scipy.signal import medfilt
    from scipy.fft import fft, ifft, next_fast_len
except ImportError:
    print("Ошибка: Библиотека scipy не установлена. Установите: pip install scipy")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Essentia теперь обязателен – всегда доступен
try:
    import essentia
    import essentia.standard as es
except ImportError:
    print("Ошибка: Библиотека essentia не установлена. Установите: pip install essentia")
    sys.exit(1)

ESSENTIA_AVAILABLE = True

whisper_available = False
try:
    import whisper

    whisper_available = True
except ImportError:
    pass

# Для записи wav в VocalAnalyzer
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==================== ЛОГГИРОВАНИЕ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== JSON ENCODER ====================
class NumpyJSONEncoder(json.JSONEncoder):
    """Безопасный энкодер для numpy типов при записи в JSON."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==================== ЦВЕТНОЙ ВЫВОД ====================
class Colorizer:
    """Потокобезопасный класс для цветного вывода."""
    _codes = {
        'RED': '\033[91m', 'GREEN': '\033[92m', 'YELLOW': '\033[93m',
        'BLUE': '\033[94m', 'MAGENTA': '\033[95m', 'CYAN': '\033[96m',
        'WHITE': '\033[97m', 'BOLD': '\033[1m', 'UNDERLINE': '\033[4m',
        'RESET': '\033[0m'
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.stdout.isatty()

    def wrap(self, text: str, *styles: str) -> str:
        if not self.enabled:
            return str(text)
        prefix = "".join(self._codes.get(s, '') for s in styles)
        return f"{prefix}{text}{self._codes['RESET']}"


# ==================== КОНСТАНТЫ ====================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_CLASS_MAP = {name: i for i, name in enumerate(NOTE_NAMES)}

# Профили тональностей
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
    'Bellman': 1.1, 'Bass Analysis': 1.2, 'Track Boundaries': 0.9,
    'Chord Voting': 1.3, 'Circle of Fifths': 0.8, 'Spectral Analysis': 0.7,
    'Essentia': 1.5,
}


# ==================== ТИПИЗАЦИЯ ====================
class MethodResult(TypedDict, total=False):
    method: str
    pitch_class: int
    key: str
    mode: str
    score: float
    confidence: float
    weight: float
    error: str


# ==================== DATACLASSES ====================
@dataclass
class AnalysisConfig:
    sample_rate: int = 44100  # Essentia использует 44100, теперь это стандарт
    trim_top_db: int = 20
    use_hpss: bool = True
    normalize_audio: bool = True
    bpm_min: int = 40
    bpm_max: int = 220
    use_whisper: bool = False
    whisper_model: str = "base"
    analyze_sections: bool = True
    analyze_rhythm: bool = True
    analyze_vocal: bool = True
    analyze_chords: bool = True
    analyze_dynamics: bool = True


@dataclass
class AudioFeatures:
    y: np.ndarray
    y_perc: np.ndarray
    sr: int
    chroma: np.ndarray
    bass_chroma: np.ndarray
    tempo: float
    beat_frames: np.ndarray
    onset_env: np.ndarray
    duration: float
    spectral_centroid: np.ndarray | None = None
    spectral_contrast: np.ndarray | None = None


@dataclass
class SectionInfo:
    label: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class ExtendedResult:
    file: str
    key: str
    mode: str
    confidence: float
    confidence_level: str
    votes: int
    total_methods: int
    bpm: dict | None = None
    duration_seconds: float = 0.0
    sample_rate: int = 22050
    structure: dict | None = None
    rhythm: dict | None = None
    vocal: dict | None = None
    chords: dict | None = None
    dynamics: dict | None = None
    all_results: list = field(default_factory=list)
    voting: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            'file': self.file,
            'duration_seconds': self.duration_seconds,
            'sample_rate': self.sample_rate,
            'key': self.key,
            'mode': self.mode,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level,
            'votes': self.votes,
            'total_methods': self.total_methods,
            'bpm': self.bpm,
            'structure': self.structure,
            'rhythm': self.rhythm,
            'vocal': self.vocal,
            'chords': self.chords,
            'dynamics': self.dynamics,
            'all_results': self.all_results,
            'voting': self.voting
        }


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def normalize_key(key_name: str) -> str:
    try:
        midi = librosa.note_to_midi(key_name)
        pc = midi % 12
        return NOTE_NAMES[pc]
    except Exception:
        mapping = {
            'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
            'Cb': 'B', 'Fb': 'E', 'B#': 'C', 'E#': 'F',
            'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'A': 'A', 'B': 'B',
            'C#': 'C#', 'D#': 'D#', 'F#': 'F#', 'G#': 'G#', 'A#': 'A#'
        }
        return mapping.get(key_name, key_name)


def normalize_audio(y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    target_peak = 10 ** (target_db / 20.0)
    return y * (target_peak / peak)


# ==================== БАЗОВЫЙ КЛАСС ДЛЯ МЕТОДОВ ====================
class KeyMethod:
    name: str = "BaseMethod"
    weight: float = 1.0

    def __init__(self, features: AudioFeatures):
        self.features = features
        self._chroma_avg = None

    @property
    def chroma_avg(self) -> np.ndarray:
        if self._chroma_avg is None:
            self._chroma_avg = np.mean(self.features.chroma, axis=1)
            self._chroma_avg = self._chroma_avg - np.mean(self._chroma_avg)
            norm = np.linalg.norm(self._chroma_avg)
            if norm > 0:
                self._chroma_avg = self._chroma_avg / norm
        return self._chroma_avg

    def detect(self) -> MethodResult:
        raise NotImplementedError

    def result(self, key_pc: int, mode: str, score: float, confidence: float) -> MethodResult:
        return {
            'method': self.name,
            'pitch_class': int(key_pc),
            'key': NOTE_NAMES[key_pc],
            'mode': mode,
            'score': float(score),
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'weight': self.weight
        }


# ==================== МЕТОДЫ 1-4: Профили ====================
class ProfileMethod(KeyMethod):
    _maj_matrices: dict = {}
    _min_matrices: dict = {}

    def __init__(self, major_profile: np.ndarray, minor_profile: np.ndarray, method_name: str, features: AudioFeatures):
        super().__init__(features)
        self.name = method_name
        self.weight = METHOD_WEIGHTS.get(self.name, 1.0)

        if method_name not in self._maj_matrices:
            maj_norm = major_profile - np.mean(major_profile)
            maj_norm /= np.linalg.norm(maj_norm)
            self._maj_matrices[method_name] = np.array([np.roll(maj_norm, i) for i in range(12)])

            min_norm = minor_profile - np.mean(minor_profile)
            min_norm /= np.linalg.norm(min_norm)
            self._min_matrices[method_name] = np.array([np.roll(min_norm, i) for i in range(12)])

        self.maj_matrix = self._maj_matrices[method_name]
        self.min_matrix = self._min_matrices[method_name]

    def detect(self) -> MethodResult:
        avg = self.chroma_avg
        maj_corrs = np.dot(self.maj_matrix, avg)
        min_corrs = np.dot(self.min_matrix, avg)

        best_maj_idx = int(np.argmax(maj_corrs))
        best_min_idx = int(np.argmax(min_corrs))

        if maj_corrs[best_maj_idx] > min_corrs[best_min_idx]:
            conf = (maj_corrs[best_maj_idx] + 1) / 2
            return self.result(best_maj_idx, 'Major', maj_corrs[best_maj_idx], conf)
        else:
            conf = (min_corrs[best_min_idx] + 1) / 2
            return self.result(best_min_idx, 'Minor', min_corrs[best_min_idx], conf)


# ==================== МЕТОД 5: Анализ баса ====================
class BassMethod(KeyMethod):
    name = "Bass Analysis"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        try:
            bass_energy = np.mean(self.features.bass_chroma, axis=1)
            tonic_idx = int(np.argmax(bass_energy))
            fifth_energy = bass_energy[(tonic_idx + 7) % 12]
            score = bass_energy[tonic_idx] + 0.5 * fifth_energy

            third_major = bass_energy[(tonic_idx + 4) % 12]
            third_minor = bass_energy[(tonic_idx + 3) % 12]
            mode = 'Major' if third_major > third_minor else 'Minor'

            confidence = min(1.0, (score / (np.max(bass_energy) + 1e-10)) * 1.2)
            return self.result(tonic_idx, mode, score, confidence)
        except Exception as e:
            return {'method': self.name, 'error': f'Bass analysis failed: {e}'}


# ==================== МЕТОД 6: Границы трека ====================
class BoundariesMethod(KeyMethod):
    name = "Track Boundaries"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        duration = self.features.duration
        window = min(10, duration * 0.1)

        if window < 0.5:
            return {'method': self.name, 'error': 'Track too short'}

        samples_frames = librosa.time_to_frames(window, sr=self.features.sr)
        chroma = self.features.chroma

        if samples_frames * 2 >= chroma.shape[1]:
            samples_frames = chroma.shape[1] // 4

        start_chroma = chroma[:, :samples_frames]
        end_chroma = chroma[:, -samples_frames:]

        boundary_chroma = (np.mean(start_chroma, axis=1) + np.mean(end_chroma, axis=1)) / 2
        boundary_chroma -= np.mean(boundary_chroma)

        tonic_idx = int(np.argmax(boundary_chroma))
        major_energy = boundary_chroma[(tonic_idx + 4) % 12]
        minor_energy = boundary_chroma[(tonic_idx + 3) % 12]

        mode = 'Major' if major_energy > minor_energy else 'Minor'
        confidence = min(1.0, abs(major_energy - minor_energy) / (abs(major_energy) + abs(minor_energy) + 1e-10) * 2)

        return self.result(tonic_idx, mode, boundary_chroma[tonic_idx], confidence)


# ==================== МЕТОД 7: Голосование по тактам ====================
class ChordVotingMethod(KeyMethod):
    name = "Chord Voting"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        beat_frames = self.features.beat_frames
        if len(beat_frames) < 8:
            return {'method': self.name, 'error': 'Not enough beats'}

        step = 4 if len(beat_frames) > 16 else 2
        tonic_votes = np.zeros(12, dtype=int)
        mode_votes = {'Major': 0, 'Minor': 0}
        chroma = self.features.chroma

        for i in range(0, len(beat_frames) - step, step):
            start_frame = beat_frames[i]
            end_frame = min(beat_frames[i + step], chroma.shape[1])
            if start_frame >= end_frame:
                continue

            segment_chroma = chroma[:, start_frame:end_frame]
            avg_chroma = np.mean(segment_chroma, axis=1)

            pitch_idx = int(np.argmax(avg_chroma))
            tonic_votes[pitch_idx] += 1

            third_major = avg_chroma[(pitch_idx + 4) % 12]
            third_minor = avg_chroma[(pitch_idx + 3) % 12]
            fifth = avg_chroma[(pitch_idx + 7) % 12]

            if third_major > third_minor and fifth > 0.3 * avg_chroma[pitch_idx]:
                mode_votes['Major'] += 1
            elif third_minor > third_major and fifth > 0.3 * avg_chroma[pitch_idx]:
                mode_votes['Minor'] += 1
            else:
                mode_votes['Major' if third_major > third_minor else 'Minor'] += 1

        if np.sum(tonic_votes) == 0:
            return {'method': self.name, 'error': 'No votes'}

        most_common = int(np.argmax(tonic_votes))
        total_tonic = np.sum(tonic_votes)
        mode = 'Major' if mode_votes['Major'] >= mode_votes['Minor'] else 'Minor'
        confidence = mode_votes[mode] / (mode_votes['Major'] + mode_votes['Minor'] + 1e-10)

        return self.result(most_common, mode, tonic_votes[most_common] / total_tonic, confidence)


# ==================== МЕТОД 8: Квинтовый круг ====================
class CircleOfFifthsMethod(KeyMethod):
    name = "Circle of Fifths"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        avg = self.chroma_avg
        indices = np.arange(12)

        major_scores = avg[indices] * 2.0 + avg[(indices + 7) % 12] * 1.5 + avg[(indices + 5) % 12] * 1.0
        minor_scores = avg[indices] * 2.0 + avg[(indices + 7) % 12] * 1.5 + avg[(indices + 3) % 12] * 1.0

        best_maj_idx = int(np.argmax(major_scores))
        best_min_idx = int(np.argmax(minor_scores))

        if major_scores[best_maj_idx] > minor_scores[best_min_idx]:
            conf = min(1.0, major_scores[best_maj_idx] / (np.max(avg) * 3 + 1e-10))
            return self.result(best_maj_idx, 'Major', major_scores[best_maj_idx], conf)
        else:
            conf = min(1.0, minor_scores[best_min_idx] / (np.max(avg) * 3 + 1e-10))
            return self.result(best_min_idx, 'Minor', minor_scores[best_min_idx], conf)


# ==================== МЕТОД 9: Спектральный анализ ====================
class SpectralMethod(KeyMethod):
    name = "Spectral Analysis"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        avg = self.chroma_avg
        tonic_idx = int(np.argmax(avg))

        centroids = self.features.spectral_centroid
        contrast = self.features.spectral_contrast

        third_diff = avg[(tonic_idx + 4) % 12] - avg[(tonic_idx + 3) % 12]

        mean_centroid = np.mean(centroids) if centroids is not None else 0
        mean_contrast = np.mean(contrast[1:]) if contrast is not None and contrast.shape[0] > 1 else 0

        if mean_centroid > 1500 and mean_contrast > 0.5:
            mode = 'Major' if third_diff > 0 else 'Minor'
        else:
            mode = 'Minor' if third_diff < 0 else 'Major'

        confidence = min(1.0, abs(third_diff) * 3 + 0.2)
        return self.result(tonic_idx, mode, avg[tonic_idx], confidence)


# ==================== МЕТОД 10: Essentia ====================
class EssentiaMethod(KeyMethod):
    name = "Essentia"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> MethodResult:
        try:
            audio = self.features.y.astype(np.float32)
            profiles = ['temperley', 'krumhansl', 'edma']
            best_result = None
            best_strength = 0.0

            for profile in profiles:
                try:
                    key_extractor = es.KeyExtractor(profileType=profile)
                    key, scale, strength = key_extractor(audio)
                    if strength > best_strength and key and scale:
                        best_strength = strength
                        best_result = (key, scale, strength)
                except Exception:
                    continue

            if not best_result:
                return {'method': self.name, 'error': 'Key extraction failed'}

            key, scale, strength = best_result
            mode = scale.capitalize()
            key_norm = normalize_key(key)

            if key_norm not in PITCH_CLASS_MAP:
                return {'method': self.name, 'error': f'Unknown key: {key}'}

            return self.result(PITCH_CLASS_MAP[key_norm], mode, strength, strength)
        except Exception as e:
            return {'method': self.name, 'error': str(e)}


# ==================== ОПРЕДЕЛЕНИЕ BPM ====================
class BPMDetector:
    @staticmethod
    def _autocorrelate_bpm_fft(signal: np.ndarray, sr: int, min_bpm: int = 40, max_bpm: int = 220) -> float | None:
        if len(signal) < sr * 2:
            return None

        signal = signal - np.mean(signal)
        n = len(signal)
        fast_n = next_fast_len(2 * n)
        fft_signal = fft(signal, n=fast_n)
        fft_autocorr = ifft(fft_signal * np.conj(fft_signal))
        autocorr = np.real(fft_autocorr[:n])

        min_lag = int(60 * sr / max_bpm)
        max_lag = int(60 * sr / min_bpm)
        if max_lag >= len(autocorr) or min_lag >= max_lag:
            return None

        search = autocorr[min_lag:max_lag]
        if len(search) == 0:
            return None

        peak_lag = np.argmax(search) + min_lag
        bpm = 60 * sr / peak_lag
        return float(bpm) if min_bpm < bpm < max_bpm else None

    @staticmethod
    def detect(features: AudioFeatures, config: AnalysisConfig) -> dict:
        results = {}
        y = features.y_perc
        sr = features.sr

        if features.tempo:
            results['librosa'] = float(features.tempo)

        try:
            envelope = np.mean(np.abs(librosa.stft(y)), axis=0)
            bpm = BPMDetector._autocorrelate_bpm_fft(envelope, sr, config.bpm_min, config.bpm_max)
            if bpm:
                results['autocorrelation'] = bpm
        except Exception:
            pass

        try:
            spectral_flux = np.diff(np.mean(np.abs(librosa.stft(y)), axis=0))
            spectral_flux = np.maximum(spectral_flux, 0)
            bpm = BPMDetector._autocorrelate_bpm_fft(spectral_flux, sr, config.bpm_min, config.bpm_max)
            if bpm:
                results['spectral_flux'] = bpm
        except Exception:
            pass

        try:
            bpm = BPMDetector._autocorrelate_bpm_fft(features.onset_env, sr, config.bpm_min, config.bpm_max)
            if bpm:
                results['onset'] = bpm
        except Exception:
            pass

        if results:
            values = list(results.values())
            median = np.median(values)
            filtered = {k: v for k, v in results.items() if abs(v - median) / median < 0.15}
            return filtered if filtered else results
        return {}


# ==================== АНАЛИЗАТОР СТРУКТУРЫ ====================
class SectionAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn не установлен', 'sections': []}

        try:
            chroma = self.features.chroma
            sr = self.features.sr
            hop_length = 512
            segment_frames = int(2 * sr / hop_length)

            if segment_frames >= chroma.shape[1]:
                return {'sections': [], 'structure_map': 'Too short', 'chorus_count': 0}

            segments = []
            times = []

            for i in range(0, chroma.shape[1] - segment_frames, segment_frames):
                seg_chroma = chroma[:, i:i + segment_frames]
                seg_features = np.mean(seg_chroma, axis=1)
                segments.append(seg_features)
                start_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i + segment_frames, sr=sr, hop_length=hop_length)
                times.append((start_time, end_time))

            if len(segments) < 3:
                return {'sections': [], 'structure_map': 'Too short', 'chorus_count': 0}

            segments = np.array(segments)
            n_clusters = min(6, len(segments))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(segments)

            # Предрасчет сумм энергий для всех сегментов
            all_seg_energies = np.array([np.sum(s) for s in segments])
            percentile_70 = np.percentile(all_seg_energies, 70)
            percentile_30 = np.percentile(all_seg_energies, 30)

            sections = []
            current_label = labels[0]
            current_start = times[0][0]

            for i in range(1, len(labels)):
                if labels[i] != current_label or i == len(labels) - 1:
                    end_idx = i if labels[i] != current_label else i
                    current_end = times[end_idx - 1][1]

                    seg_indices = [j for j in range(len(labels)) if labels[j] == current_label]
                    seg_energy = np.mean([all_seg_energies[j] for j in seg_indices])

                    if current_start < 5:
                        label = "Intro"
                    elif current_end > self.features.duration - 5:
                        label = "Outro"
                    elif seg_energy > percentile_70:
                        label = "Chorus"
                    elif seg_energy < percentile_30:
                        label = "Bridge"
                    else:
                        label = "Verse"

                    sections.append(SectionInfo(
                        label=label,
                        start_time=round(current_start, 2),
                        end_time=round(current_end, 2),
                        duration=round(current_end - current_start, 2)
                    ))

                    current_label = labels[i]
                    current_start = times[i][0]

            chorus_count = sum(1 for s in sections if s.label == "Chorus")
            structure_map = " → ".join([s.label for s in sections])

            return {
                'sections': [
                    {'label': s.label, 'start_time': s.start_time,
                     'end_time': s.end_time, 'duration': s.duration}
                    for s in sections
                ],
                'structure_map': structure_map,
                'chorus_count': chorus_count,
                'total_sections': len(sections)
            }

        except Exception as e:
            logger.error(f"Section analysis failed: {e}")
            return {'error': str(e), 'sections': []}


# ==================== АНАЛИЗАТОР РИТМА ====================
class RhythmAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            tempo = self.features.tempo
            beat_frames = self.features.beat_frames
            sr = self.features.sr

            if tempo <= 0 or len(beat_frames) < 4:
                return {'error': 'Cannot determine rhythm'}

            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                intervals = np.diff(beat_times)
                median_interval = np.median(intervals)
                interval_variance = np.var(intervals) / (median_interval ** 2)

                if interval_variance < 0.01:
                    meter = "4/4"
                elif interval_variance < 0.02:
                    meter = "3/4"
                elif interval_variance < 0.03:
                    meter = "6/8"
                else:
                    meter = "Free"
            else:
                meter = "Unknown"
                interval_variance = 0

            regularity = max(0, 1 - interval_variance * 10) if len(beat_times) > 1 else 0

            if len(beat_frames) > 2:
                onset_env = self.features.onset_env
                beat_strength = np.mean([onset_env[min(b, len(onset_env) - 1)] for b in beat_frames])
                avg_strength = np.mean(onset_env)
                syncopation = max(0, 1 - (beat_strength / (avg_strength + 1e-10)))
            else:
                syncopation = 0

            return {
                'tempo_bpm': round(float(tempo), 1),
                'meter': meter,
                'beat_count': len(beat_frames),
                'regularity': round(float(regularity), 3),
                'syncopation': round(float(syncopation), 3),
                'beat_times': [round(float(t), 2) for t in beat_times[:20]]
            }

        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            return {'error': str(e)}


# ==================== АНАЛИЗАТОР ВОКАЛА ====================
class VocalAnalyzer:
    """Анализ вокала и лирики."""

    # Кэшируем модель Whisper на уровне класса, чтобы не загружать её для каждого файла
    _whisper_model_cache: dict = {}

    def __init__(self, features: AudioFeatures, use_whisper: bool = False, whisper_model: str = "base"):
        self.features = features
        self.use_whisper = use_whisper and whisper_available and SOUNDFILE_AVAILABLE
        self.whisper_model_name = whisper_model

    def _get_whisper_model(self):
        if self.whisper_model_name not in self._whisper_model_cache:
            self._whisper_model_cache[self.whisper_model_name] = whisper.load_model(self.whisper_model_name)
        return self._whisper_model_cache[self.whisper_model_name]

    def analyze(self) -> dict:
        try:
            y = self.features.y
            sr = self.features.sr
            hop_length = 512

            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)

            vocal_band = (freqs >= 200) & (freqs <= 2000)
            vocal_energy = np.mean(S[vocal_band, :], axis=0)
            total_energy = np.mean(S, axis=0)

            vocal_ratio = vocal_energy / (total_energy + 1e-10)
            vocal_presence = np.mean(vocal_ratio > 0.3)

            vocal_segments = []
            threshold = 0.3
            in_vocal = False
            start_time = 0

            for i, ratio in enumerate(vocal_ratio):
                time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                if ratio > threshold and not in_vocal:
                    in_vocal = True
                    start_time = time
                elif ratio <= threshold and in_vocal:
                    in_vocal = False
                    if time - start_time > 0.5:
                        vocal_segments.append({
                            'start': round(start_time, 2),
                            'end': round(time, 2),
                            'duration': round(time - start_time, 2)
                        })

            if in_vocal:
                end_time = librosa.frames_to_time(len(vocal_ratio), sr=sr, hop_length=hop_length)
                if end_time - start_time > 0.5:
                    vocal_segments.append({
                        'start': round(start_time, 2),
                        'end': round(end_time, 2),
                        'duration': round(end_time - start_time, 2)
                    })

            result = {
                'vocal_presence_percent': round(float(vocal_presence * 100), 1),
                'vocal_segments': vocal_segments,
                'vocal_segments_count': len(vocal_segments)
            }

            if self.use_whisper:
                try:
                    model = self._get_whisper_model()
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        sf.write(tmp.name, y, sr)
                        transcription = model.transcribe(tmp.name)
                    Path(tmp.name).unlink(missing_ok=True)

                    text = transcription.get("text", "")
                    language = transcription.get("language", "unknown")

                    words = text.lower().split()
                    phrase_counter = Counter()
                    for n in [2, 3]:
                        for i in range(len(words) - n + 1):
                            phrase = " ".join(words[i:i + n])
                            phrase_counter[phrase] += 1

                    repeated_phrases = [
                        {'phrase': phrase, 'count': count}
                        for phrase, count in phrase_counter.most_common(10)
                        if count > 1
                    ]

                    result.update({
                        'transcription': text[:500],
                        'language': language,
                        'repeated_phrases': repeated_phrases,
                        'whisper_used': True
                    })

                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
                    result['whisper_used'] = False
                    result['whisper_error'] = str(e)

            return result

        except Exception as e:
            logger.error(f"Vocal analysis failed: {e}")
            return {'error': str(e)}


# ==================== АНАЛИЗАТОР АККОРДОВ (Essentia) ====================
class ChordAnalyzer:
    """Анализ гармонической прогрессии – только через Essentia."""

    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            audio = self.features.y.astype(np.float32)
            sr = int(self.features.sr)
            hop_size = 512
            frame_size = 4096

            # Правильный пайплайн Essentia для HPCP
            frame_generator = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
            windowing = es.Windowing(type='blackmanharris62')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks(maxFrequency=5000.0, magnitudeThreshold=0.00001, minFrequency=40.0)
            hpcp_extractor = es.HPCP(sampleRate=sr, minFrequency=40.0, maxFrequency=5000.0)

            hpcp_list = []
            for frame in frame_generator:
                spec = spectrum(windowing(frame))
                freqs, mags = spectral_peaks(spec)
                hpcp = hpcp_extractor(freqs, mags)
                hpcp_list.append(hpcp)

            if not hpcp_list:
                return {'error': 'No HPCP extracted'}

            hpcp_array = np.array(hpcp_list)

            # ChordsDetection принимает 2D массив HPCP (frames x 12)
            chord_detector = es.ChordsDetection()
            chords, strengths = chord_detector(hpcp_array)

            # Группируем одинаковые подряд идущие аккорды для краткости
            chord_progression = []
            current_chord = None
            current_start_time = 0.0

            for i, chord in enumerate(chords):
                time = i * hop_size / sr
                if chord != current_chord:
                    if current_chord is not None:
                        chord_progression.append({
                            'chord': str(current_chord),
                            'start_time': round(float(current_start_time), 2),
                            'end_time': round(float(time), 2)
                        })
                    current_chord = chord
                    current_start_time = time

            # Добавляем последний аккорд
            if current_chord is not None:
                final_time = len(chords) * hop_size / sr
                chord_progression.append({
                    'chord': str(current_chord),
                    'start_time': round(float(current_start_time), 2),
                    'end_time': round(float(final_time), 2)
                })

            return {
                'method': 'Essentia',
                'chords': chord_progression,
                'chord_count': len(chord_progression),
                'progression': " - ".join([c['chord'] for c in chord_progression[:8]])
            }

        except Exception as e:
            logger.error(f"Chord analysis failed: {e}")
            return {'error': str(e)}


# ==================== АНАЛИЗАТОР ДИНАМИКИ ====================
class DynamicsAnalyzer:
    def __init__(self, features: AudioFeatures):
        self.features = features

    def analyze(self) -> dict:
        try:
            y = self.features.y
            sr = self.features.sr
            hop_length = 512

            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)

            max_db = np.max(rms_db)
            min_db = np.percentile(rms_db, 5)
            dynamic_range = max_db - min_db

            threshold_loud = np.percentile(rms_db, 70)
            threshold_quiet = np.percentile(rms_db, 30)

            loud_segments = []
            quiet_segments = []
            segment_length = int(2 * sr / hop_length)

            for i in range(0, len(rms_db), segment_length):
                segment = rms_db[i:i + segment_length]
                if len(segment) < segment_length // 2:
                    break

                segment_mean = np.mean(segment)
                start_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i + len(segment), sr=sr, hop_length=hop_length)

                if segment_mean > threshold_loud:
                    loud_segments.append({
                        'start': round(float(start_time), 2),
                        'end': round(float(end_time), 2),
                        'mean_db': round(float(segment_mean), 1)
                    })
                elif segment_mean < threshold_quiet:
                    quiet_segments.append({
                        'start': round(float(start_time), 2),
                        'end': round(float(end_time), 2),
                        'mean_db': round(float(segment_mean), 1)
                    })

            try:
                cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
                cent_mean = np.mean(cent)
                cent_std = np.std(cent)

                solo_segments = []
                for i in range(0, len(cent), segment_length):
                    segment = cent[i:i + segment_length]
                    if len(segment) < segment_length // 2:
                        break

                    if np.mean(segment) > cent_mean + cent_std and np.std(segment) > cent_std * 0.5:
                        start_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                        end_time = librosa.frames_to_time(i + len(segment), sr=sr, hop_length=hop_length)
                        solo_segments.append({
                            'start': round(float(start_time), 2),
                            'end': round(float(end_time), 2),
                            'mean_centroid': round(float(np.mean(segment)), 1)
                        })
            except Exception:
                solo_segments = []

            energy_profile = []
            profile_step = max(1, int(self.features.duration / 10))
            step_frames = len(rms_db) // profile_step

            for i in range(profile_step):
                start_idx = i * step_frames
                end_idx = min((i + 1) * step_frames, len(rms_db))
                if start_idx < len(rms_db):
                    segment_mean = np.mean(rms_db[start_idx:end_idx])
                    energy_profile.append(round(float(segment_mean), 1))

            return {
                'dynamic_range_db': round(float(dynamic_range), 1),
                'loud_segments': loud_segments,
                'loud_segments_count': len(loud_segments),
                'quiet_segments': quiet_segments,
                'quiet_segments_count': len(quiet_segments),
                'instrumental_solos': solo_segments,
                'solos_count': len(solo_segments),
                'energy_profile': energy_profile,
                'max_rms_db': round(float(max_db), 1),
                'min_rms_db': round(float(min_db), 1)
            }

        except Exception as e:
            logger.error(f"Dynamics analysis failed: {e}")
            return {'error': str(e)}


# ==================== ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА ====================
class KeyDetector:
    def __init__(self, filepath: str, config: AnalysisConfig | None = None):
        self.filepath = Path(filepath)
        self.config = config or AnalysisConfig()
        self.features: AudioFeatures | None = None
        self.methods: list[KeyMethod] = []
        self.results: list[MethodResult] = []
        self.bpm_results: dict = {}
        self.final_result: ExtendedResult | None = None

    def _load_audio(self) -> tuple[np.ndarray, np.ndarray, int, float]:
        logger.info(f"Loading audio: {self.filepath}")
        try:
            y, sr = librosa.load(str(self.filepath), sr=44100, mono=True)

            if len(y) == 0:
                raise ValueError("Audio file is empty or silent")

            y, _ = librosa.effects.trim(y, top_db=self.config.trim_top_db)

            if self.config.normalize_audio:
                y = normalize_audio(y)

            if self.config.use_hpss:
                y_harm, y_perc = librosa.effects.hpss(y)
            else:
                y_harm, y_perc = y, np.zeros_like(y)

            duration = len(y) / sr
            logger.info(f"Duration: {duration:.1f}s, SR: {sr}Hz")
            return y_harm, y_perc, sr, duration

        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def _extract_features(self, y: np.ndarray, y_perc: np.ndarray, sr: int, duration: float) -> AudioFeatures:
        logger.info("Extracting global features...")
        hop_length = 512

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=36)
        chroma = medfilt(chroma, kernel_size=(1, 3))

        fmin_bass = librosa.note_to_hz('C1')
        bass_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=36, fmin=fmin_bass,
                                                 n_octaves=2)

        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
            tempo = float(np.atleast_1d(tempo).mean())
        except Exception:
            tempo, beat_frames = 0.0, np.array([])

        onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)

        try:
            cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        except Exception:
            cent = None
        try:
            contr = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        except Exception:
            contr = None

        return AudioFeatures(
            y=y, y_perc=y_perc, sr=sr, chroma=chroma, bass_chroma=bass_chroma,
            tempo=tempo, beat_frames=beat_frames, onset_env=onset_env,
            duration=duration, spectral_centroid=cent, spectral_contrast=contr
        )

    def _init_methods(self):
        self.methods = [
            ProfileMethod(KRUMHANSL_MAJOR, KRUMHANSL_MINOR, "Krumhansl-Schmuckler", self.features),
            ProfileMethod(TEMPERLEY_MAJOR, TEMPERLEY_MINOR, "Temperley", self.features),
            ProfileMethod(ALBRECHT_MAJOR, ALBRECHT_MINOR, "Albrecht-Shanahan", self.features),
            ProfileMethod(BELLMAN_MAJOR, BELLMAN_MINOR, "Bellman", self.features),
            BassMethod(self.features),
            BoundariesMethod(self.features),
            ChordVotingMethod(self.features),
            CircleOfFifthsMethod(self.features),
            SpectralMethod(self.features),
            EssentiaMethod(self.features)
        ]

    def _run_extended_analysis(self) -> tuple[dict | None, dict | None, dict | None, dict | None, dict | None]:
        structure = rhythm = vocal = chords = dynamics = None

        if self.config.analyze_sections:
            logger.info("Analyzing song structure...")
            analyzer = SectionAnalyzer(self.features)
            structure = analyzer.analyze()

        if self.config.analyze_rhythm:
            logger.info("Analyzing rhythm...")
            analyzer = RhythmAnalyzer(self.features)
            rhythm = analyzer.analyze()

        if self.config.analyze_vocal:
            logger.info("Analyzing vocals...")
            analyzer = VocalAnalyzer(
                self.features,
                use_whisper=self.config.use_whisper,
                whisper_model=self.config.whisper_model
            )
            vocal = analyzer.analyze()

        if self.config.analyze_chords:
            logger.info("Analyzing chords...")
            analyzer = ChordAnalyzer(self.features)
            chords = analyzer.analyze()

        if self.config.analyze_dynamics:
            logger.info("Analyzing dynamics...")
            analyzer = DynamicsAnalyzer(self.features)
            dynamics = analyzer.analyze()

        return structure, rhythm, vocal, chords, dynamics

    def run(self) -> ExtendedResult:
        try:
            y, y_perc, sr, duration = self._load_audio()
            self.features = self._extract_features(y, y_perc, sr, duration)

            del y, y_perc

            self._init_methods()
            logger.info("Running key detection methods...")
            self.results = []

            for method in self.methods:
                try:
                    res = method.detect()
                    if 'error' in res:
                        logger.debug(f"{method.name}: {res['error']}")
                    else:
                        logger.info(f"{method.name}: {res['key']} {res['mode']} (conf: {res['confidence']:.2f})")
                    self.results.append(res)
                except Exception as e:
                    logger.warning(f"Method {method.name} failed: {e}")
                    self.results.append({'method': method.name, 'error': str(e)})

            logger.info("Detecting BPM...")
            self.bpm_results = BPMDetector.detect(self.features, self.config)
            for k, v in self.bpm_results.items():
                logger.info(f"BPM {k}: {v:.1f}")

            structure, rhythm, vocal, chords, dynamics = self._run_extended_analysis()

            self.final_result = self._aggregate_results(
                structure=structure, rhythm=rhythm, vocal=vocal,
                chords=chords, dynamics=dynamics
            )

            # Освобождаем память
            self.features.y = None
            self.features.y_perc = None
            self.features.chroma = None
            self.features.bass_chroma = None
            self.features = None

            return self.final_result
        except Exception as e:
            return ExtendedResult(
                file=str(self.filepath), key='N/A', mode='N/A', confidence=0.0,
                confidence_level='ОШИБКА', votes=0, total_methods=0, error=str(e)
            )

    def _aggregate_results(self, structure=None, rhythm=None, vocal=None, chords=None, dynamics=None) -> ExtendedResult:
        valid = [r for r in self.results if 'error' not in r and 'pitch_class' in r]
        if not valid:
            return ExtendedResult(
                file=str(self.filepath), key='N/A', mode='N/A', confidence=0.0,
                confidence_level='ОШИБКА', votes=0, total_methods=0, error='No valid results'
            )

        groups = defaultdict(lambda: {'votes': 0, 'weighted_sum': 0.0, 'methods': []})
        for res in valid:
            group_key = (res['pitch_class'], res['mode'])
            groups[group_key]['votes'] += 1
            groups[group_key]['weighted_sum'] += res['confidence'] * res.get('weight', 1.0)
            groups[group_key]['methods'].append(res['method'])

        sorted_groups = sorted(groups.items(),
                               key=lambda x: (x[1]['votes'], x[1]['weighted_sum']),
                               reverse=True)

        winner_pc, winner_mode = sorted_groups[0][0]
        winner_data = sorted_groups[0][1]
        total_weighted = sum(g['weighted_sum'] for _, g in groups.items())

        confidence = winner_data['weighted_sum'] / total_weighted if total_weighted > 0 else 0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        level = "ОЧЕНЬ ВЫСОКАЯ" if confidence > 0.8 else "ВЫСОКАЯ" if confidence > 0.6 else "СРЕДНЯЯ" if confidence > 0.4 else "НИЗКАЯ"
        winner_key = NOTE_NAMES[winner_pc]

        bpm_dict = None
        if self.bpm_results:
            bpm_vals = list(self.bpm_results.values())
            median_bpm = float(np.median(bpm_vals))
            bpm_dict = {
                'average': float(np.mean(bpm_vals)),
                'median': median_bpm,
                'rounded': int(round(median_bpm)),
                'range': [float(np.min(bpm_vals)), float(np.max(bpm_vals))],
                'methods': {k: float(v) for k, v in self.bpm_results.items()}
            }

        voting = {}
        for (pc, mode), data in sorted_groups:
            key_str = f"{NOTE_NAMES[pc]} {mode}"
            voting[key_str] = {
                'votes': data['votes'],
                'total_weighted_score': round(data['weighted_sum'], 4),
                'avg_weighted_confidence': round(data['weighted_sum'] / data['votes'], 4),
                'methods': data['methods']
            }

        return ExtendedResult(
            file=str(self.filepath), key=winner_key, mode=winner_mode, confidence=confidence,
            confidence_level=level, votes=winner_data['votes'], total_methods=len(valid),
            bpm=bpm_dict, duration_seconds=self.features.duration if self.features else 0.0,
            sample_rate=self.features.sr if self.features else 0,
            structure=structure, rhythm=rhythm, vocal=vocal, chords=chords, dynamics=dynamics,
            all_results=self.results, voting=voting
        )

    def save_json(self, output_path: Path | None = None):
        if not self.final_result:
            logger.warning("No results to save.")
            return

        if output_path is None:
            output_path = self.filepath.with_name(self.filepath.stem + '_analysis.json')
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_result.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        logger.info(f"Results saved to {output_path}")


# ==================== ПАКЕТНАЯ ОБРАБОТКА ====================
def format_result_output(result: ExtendedResult, colorizer: Colorizer) -> str:
    if result.error:
        return colorizer.wrap(f"❌ Error [{result.file}]: {result.error}", 'RED')

    lines = [
        f"\n{colorizer.wrap('=' * 70, 'CYAN')}",
        colorizer.wrap(f" 🎵 ФИНАЛЬНЫЙ ВЕРДИКТ: {Path(result.file).name}", 'BOLD', 'GREEN'),
        colorizer.wrap('=' * 70, 'CYAN'),
        f"  {colorizer.wrap('Тональность:', 'BOLD')} {colorizer.wrap(f'{result.key} {result.mode}', 'YELLOW')}",
        f"  {colorizer.wrap('Уверенность:', 'BOLD')} {result.confidence:.1%} ({result.confidence_level})",
        f"  {colorizer.wrap('Согласованность:', 'BOLD')} {result.votes}/{result.total_methods} методов"
    ]

    if result.bpm:
        bpm = result.bpm
        lines.append(
            f"\n  {colorizer.wrap('BPM:', 'BOLD')} среднее {bpm['average']:.1f}, медиана {bpm['median']:.1f}, округлённое {bpm['rounded']}")
        lines.append(f"  {colorizer.wrap('Диапазон:', 'BOLD')} {bpm['range'][0]:.1f} – {bpm['range'][1]:.1f}")

    if result.structure and 'error' not in result.structure:
        lines.append(f"\n{colorizer.wrap('📊 СТРУКТУРА:', 'BOLD')}")
        lines.append(f"    Секций: {result.structure.get('total_sections', 0)}")
        lines.append(f"    Припевов: {result.structure.get('chorus_count', 0)}")
        lines.append(f"    Структура: {result.structure.get('structure_map', 'N/A')}")
    elif result.structure and 'error' in result.structure:
        lines.append(
            f"\n{colorizer.wrap('📊 СТРУКТУРА:', 'BOLD')} {colorizer.wrap(result.structure['error'], 'YELLOW')}")

    if result.rhythm and 'error' not in result.rhythm:
        lines.append(f"\n{colorizer.wrap('🎼 РИТМ:', 'BOLD')}")
        lines.append(f"    Размер: {result.rhythm.get('meter', 'N/A')}")
        lines.append(f"    Регулярность: {result.rhythm.get('regularity', 0):.1%}")
        lines.append(f"    Синкопирование: {result.rhythm.get('syncopation', 0):.2f}")
    elif result.rhythm and 'error' in result.rhythm:
        lines.append(f"\n{colorizer.wrap('🎼 РИТМ:', 'BOLD')} {colorizer.wrap(result.rhythm['error'], 'YELLOW')}")

    if result.vocal and 'error' not in result.vocal:
        lines.append(f"\n{colorizer.wrap('🎤 ВОКАЛ:', 'BOLD')}")
        lines.append(f"    Присутствие вокала: {result.vocal.get('vocal_presence_percent', 0):.1f}%")
        if 'language' in result.vocal:
            lines.append(f"    Язык: {result.vocal['language']}")
        if 'repeated_phrases' in result.vocal and result.vocal['repeated_phrases']:
            lines.append(f"    Повторяющиеся фразы: {len(result.vocal['repeated_phrases'])}")
            for phrase_info in result.vocal['repeated_phrases'][:3]:
                lines.append(f"      • \"{phrase_info['phrase']}\" ({phrase_info['count']}x)")
    elif result.vocal and 'error' in result.vocal:
        lines.append(f"\n{colorizer.wrap('🎤 ВОКАЛ:', 'BOLD')} {colorizer.wrap(result.vocal['error'], 'YELLOW')}")

    if result.chords and 'error' not in result.chords:
        lines.append(f"\n{colorizer.wrap('🎸 АККОРДЫ:', 'BOLD')}")
        lines.append(f"    Метод: {result.chords.get('method', 'N/A')}")
        lines.append(f"    Аккордов: {result.chords.get('chord_count', 0)}")
        if 'progression' in result.chords:
            lines.append(f"    Прогрессия: {result.chords['progression']}")
    elif result.chords and 'error' in result.chords:
        lines.append(f"\n{colorizer.wrap('🎸 АККОРДЫ:', 'BOLD')} {colorizer.wrap(result.chords['error'], 'YELLOW')}")

    if result.dynamics and 'error' not in result.dynamics:
        lines.append(f"\n{colorizer.wrap('📈 ДИНАМИКА:', 'BOLD')}")
        lines.append(f"    Динамический диапазон: {result.dynamics.get('dynamic_range_db', 0):.1f} dB")
        lines.append(f"    Громких секций: {result.dynamics.get('loud_segments_count', 0)}")
        lines.append(f"    Тихих секций: {result.dynamics.get('quiet_segments_count', 0)}")
        lines.append(f"    Инструментальных соло: {result.dynamics.get('solos_count', 0)}")
    elif result.dynamics and 'error' in result.dynamics:
        lines.append(f"\n{colorizer.wrap('📈 ДИНАМИКА:', 'BOLD')} {colorizer.wrap(result.dynamics['error'], 'YELLOW')}")

    if result.confidence < 0.6:
        lines.append(
            f"\n{colorizer.wrap('💡 Рекомендация: Essentia уже используется. Для повышения точности проверьте качество аудио.', 'YELLOW')}")

    lines.append(colorizer.wrap('=' * 70, 'CYAN'))
    return "\n".join(lines)


def process_file(filepath: str, config: AnalysisConfig, save: bool, output_dir: str | None) -> ExtendedResult:
    path = Path(filepath)
    if not path.exists():
        return ExtendedResult(file=filepath, key='N/A', mode='N/A', confidence=0.0,
                              confidence_level='ОШИБКА', votes=0, total_methods=0,
                              error='File not found')

    try:
        detector = KeyDetector(filepath, config=config)
        result = detector.run()

        if save and not result.error:
            out_path = Path(output_dir) / (path.stem + '_analysis.json') if output_dir else None
            detector.save_json(out_path)

        return result
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return ExtendedResult(file=filepath, key='N/A', mode='N/A', confidence=0.0,
                              confidence_level='ОШИБКА', votes=0, total_methods=0,
                              error=str(e))


def get_audio_files(paths: list[str], recursive: bool = False) -> list[str]:
    valid_exts = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            glob_method = path.rglob if recursive else path.glob
            files.extend([str(f) for f in glob_method('*') if f.suffix.lower() in valid_exts])
        elif path.is_file() and path.suffix.lower() in valid_exts:
            files.append(str(path))
    return files


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description="Продвинутый анализатор тональности, BPM, структуры, ритма, вокала, аккордов и динамики",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s song.mp3
  %(prog)s song.mp3 --whisper --whisper-model medium
  %(prog)s *.mp3 --output results/ --parallel 4
  %(prog)s song.wav --no-save --verbose --json
  %(prog)s ./music_folder/ --recursive
  %(prog)s song.mp3 --no-sections --no-rhythm --no-vocal --no-chords --no-dynamics
        """
    )
    parser.add_argument('files', nargs='*', help='Путь к аудиофайлу(ам) или папкам')
    parser.add_argument('--output', '-o', help='Папка для сохранения JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять JSON файлы')
    parser.add_argument('--json', action='store_true', help='Выводить только JSON в консоль')
    parser.add_argument('--parallel', '-p', type=int, default=1, help='Количество параллельных процессов')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--no-color', action='store_true', help='Отключить цветной вывод')
    parser.add_argument('--no-hpss', action='store_true', help='Отключить HPSS разделение')
    parser.add_argument('--no-normalize', action='store_true', help='Отключить нормализацию громкости')
    parser.add_argument('--recursive', '-r', action='store_true', help='Рекурсивный поиск аудио в папках')

    parser.add_argument('--whisper', action='store_true', help='Использовать Whisper для транскрибации вокала')
    parser.add_argument('--whisper-model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Модель Whisper (по умолчанию: base)')
    parser.add_argument('--no-sections', action='store_true', help='Отключить анализ структуры')
    parser.add_argument('--no-rhythm', action='store_true', help='Отключить анализ ритма')
    parser.add_argument('--no-vocal', action='store_true', help='Отключить анализ вокала')
    parser.add_argument('--no-chords', action='store_true', help='Отключить анализ аккордов')
    parser.add_argument('--no-dynamics', action='store_true', help='Отключить анализ динамики')

    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    use_color = not args.no_color
    colorizer = Colorizer(enabled=use_color)

    config = AnalysisConfig(
        use_hpss=not args.no_hpss,
        normalize_audio=not args.no_normalize,
        use_whisper=args.whisper,
        whisper_model=args.whisper_model,
        analyze_sections=not args.no_sections,
        analyze_rhythm=not args.no_rhythm,
        analyze_vocal=not args.no_vocal,
        analyze_chords=not args.no_chords,
        analyze_dynamics=not args.no_dynamics
    )

    files_to_process = get_audio_files(args.files, recursive=args.recursive)

    if not files_to_process:
        print(colorizer.wrap("Не найдено аудиофайлов для обработки.", 'RED'))
        return

    if args.json:
        logging.disable(logging.CRITICAL)

    if args.parallel <= 1 or len(files_to_process) == 1:
        for filepath in tqdm(files_to_process, desc="Обработка файлов", file=sys.stdout,
                             dynamic_ncols=True, disable=args.json):
            result = process_file(filepath, config, not args.no_save, args.output)
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, cls=NumpyJSONEncoder))
            else:
                tqdm.write(format_result_output(result, colorizer))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_file, filepath, config, not args.no_save, args.output): filepath
                for filepath in files_to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка файлов",
                               file=sys.stdout, dynamic_ncols=True, disable=args.json):
                try:
                    result = future.result()
                    if args.json:
                        print(json.dumps(result.to_dict(), ensure_ascii=False, cls=NumpyJSONEncoder))
                    else:
                        tqdm.write(format_result_output(result, colorizer))
                except Exception as e:
                    if not args.json:
                        tqdm.write(colorizer.wrap(f"❌ Критическая ошибка потока: {e}", 'RED'))


if __name__ == "__main__":
    main()
