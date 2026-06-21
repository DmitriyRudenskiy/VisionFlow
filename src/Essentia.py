#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Максимально точное определение тональности и BPM (улучшенная версия).
Поддерживает пакетную обработку, экспорт в JSON и ансамблевое голосование методов.
"""

import numpy as np
import json
import os
import argparse
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==================== ЛОГГИРОВАНИЕ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== КОНСТАНТЫ ====================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_CLASS_MAP = {name: i for i, name in enumerate(NOTE_NAMES)}

# Профили Krumhansl-Schmuckler и Temperley
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.0, 2.0, 4.0, 4.0, 2.0, 5.0, 2.0, 3.0, 2.0, 3.0])
TEMPERLEY_MINOR = np.array([5.0, 3.0, 2.0, 4.0, 2.0, 3.0, 2.0, 5.0, 3.0, 2.0, 4.0, 2.0])

METHOD_WEIGHTS = {
    'Krumhansl-Schmuckler': 1.0,
    'Temperley': 1.2,
    'Bass Analysis': 1.1,
    'Track Boundaries': 0.9,
    'Chord Voting': 1.3,
    'Circle of Fifths': 0.8,
    'Spectral Analysis': 0.7,
    'Essentia': 1.5,
}


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def normalize_key(key_name: str) -> str:
    """Приводит энгармонические названия к диезным (Eb → D#, Bb → A#, etc.)."""
    mapping = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
        'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'A': 'A', 'B': 'B',
        'C#': 'C#', 'D#': 'D#', 'F#': 'F#', 'G#': 'G#', 'A#': 'A#'
    }
    return mapping.get(key_name, key_name)


def pitch_class_to_key(pc: int, mode: str) -> str:
    """Обратное преобразование pc + mode в строку."""
    return f"{NOTE_NAMES[pc]} {mode}"


# ==================== ПОПЫТКА ИМПОРТА ====================
def try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


librosa = try_import('librosa')
essentia = try_import('essentia')
es = None
if essentia is not None:
    try:
        import essentia.standard as es
    except Exception as e:
        logger.warning(f"Essentia imported but standard module failed: {e}")


# ==================== БАЗОВЫЙ КЛАСС ДЛЯ МЕТОДОВ ====================
class KeyMethod:
    """Базовый класс для всех методов определения тональности."""
    name: str = "BaseMethod"
    weight: float = 1.0

    def __init__(self, y: np.ndarray, sr: int, chroma: Optional[np.ndarray] = None):
        self.y = y
        self.sr = sr
        self.chroma = chroma
        self._chroma_avg = None

    @property
    def chroma_avg(self) -> np.ndarray:
        if self._chroma_avg is None:
            if self.chroma is not None:
                self._chroma_avg = np.mean(self.chroma, axis=1)
            else:
                chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, bins_per_octave=36)
                self._chroma_avg = np.mean(chroma, axis=1)
            # Нормализация (удаление среднего)
            self._chroma_avg = self._chroma_avg - np.mean(self._chroma_avg)
        return self._chroma_avg

    def detect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def result(self, key_pc: int, mode: str, score: float, confidence: float) -> Dict:
        return {
            'method': self.name,
            'pitch_class': int(key_pc),
            'key': NOTE_NAMES[key_pc],
            'mode': mode,
            'score': float(score),
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'weight': self.weight
        }


# ==================== МЕТОД 1 & 2: Профили (Krumhansl / Temperley) ====================
class ProfileMethod(KeyMethod):
    """Универсальный метод для тональных профилей."""

    def __init__(self, major_profile: np.ndarray, minor_profile: np.ndarray, method_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = method_name
        self.weight = METHOD_WEIGHTS.get(self.name, 1.0)
        self.major_profile = major_profile
        self.minor_profile = minor_profile

    def detect(self) -> Dict:
        avg = self.chroma_avg
        best_score = -np.inf
        best_key = 0
        best_mode = 'Major'

        norm_avg = np.linalg.norm(avg)

        for i in range(12):
            maj_prof = np.roll(self.major_profile, i)
            min_prof = np.roll(self.minor_profile, i)

            maj_corr = np.dot(avg, maj_prof) / (norm_avg * np.linalg.norm(maj_prof) + 1e-10)
            min_corr = np.dot(avg, min_prof) / (norm_avg * np.linalg.norm(min_prof) + 1e-10)

            if maj_corr > best_score:
                best_score, best_key, best_mode = maj_corr, i, 'Major'
            if min_corr > best_score:
                best_score, best_key, best_mode = min_corr, i, 'Minor'

        confidence = (best_score + 1) / 2  # Перевод корреляции [-1, 1] в [0, 1]
        return self.result(best_key, best_mode, best_score, confidence)


# ==================== МЕТОД 3: Анализ баса ====================
class BassMethod(KeyMethod):
    name = "Bass Analysis"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> Dict:
        try:
            from scipy.signal import butter, lfilter
            # Настоящий low-pass фильтр для изоляции баса
            nyq = 0.5 * self.sr
            b, a = butter(4, 150.0 / nyq, btype='low')
            y_bass = lfilter(b, a, self.y)

            bass_chroma = librosa.feature.chroma_cqt(y=y_bass, sr=self.sr, bins_per_octave=36)
            bass_energy = np.mean(bass_chroma, axis=1)

            tonic_idx = int(np.argmax(bass_energy))
            fifth_energy = bass_energy[(tonic_idx + 7) % 12]
            score = bass_energy[tonic_idx] + 0.5 * fifth_energy

            third_major = bass_energy[(tonic_idx + 4) % 12]
            third_minor = bass_energy[(tonic_idx + 3) % 12]
            mode = 'Major' if third_major > third_minor else 'Minor'

            confidence = min(1.0, (score / (np.max(bass_energy) + 1e-10)) * 1.2)
            return self.result(tonic_idx, mode, score, confidence)
        except Exception as e:
            return {'method': self.name, 'error': f'Bass filter failed: {e}'}


# ==================== МЕТОД 4: Границы трека ====================
class BoundariesMethod(KeyMethod):
    name = "Track Boundaries"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> Dict:
        duration = len(self.y) / self.sr
        window = min(10, duration * 0.1)
        samples = int(window * self.sr)

        if samples < self.sr * 0.5:
            return {'method': self.name, 'error': 'Track too short'}

        start_y = self.y[:samples]
        end_y = self.y[-samples:]

        start_chroma = librosa.feature.chroma_cqt(y=start_y, sr=self.sr, bins_per_octave=36)
        end_chroma = librosa.feature.chroma_cqt(y=end_y, sr=self.sr, bins_per_octave=36)

        boundary_chroma = (np.mean(start_chroma, axis=1) + np.mean(end_chroma, axis=1)) / 2
        boundary_chroma -= np.mean(boundary_chroma)

        tonic_idx = int(np.argmax(boundary_chroma))
        major_energy = boundary_chroma[(tonic_idx + 4) % 12]
        minor_energy = boundary_chroma[(tonic_idx + 3) % 12]

        mode = 'Major' if major_energy > minor_energy else 'Minor'
        confidence = min(1.0, abs(major_energy - minor_energy) / (abs(major_energy) + abs(minor_energy) + 1e-10) * 2)

        return self.result(tonic_idx, mode, boundary_chroma[tonic_idx], confidence)


# ==================== МЕТОД 5: Голосование по тактам ====================
class ChordVotingMethod(KeyMethod):
    name = "Chord Voting"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> Dict:
        try:
            tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
            if isinstance(tempo, np.ndarray): tempo = float(tempo[0])
            if len(beats) < 8:
                return {'method': self.name, 'error': 'Not enough beats'}
        except Exception as e:
            return {'method': self.name, 'error': f'Beat tracking failed: {e}'}

        beat_frames = librosa.frames_to_samples(beats)
        step = 4 if len(beats) > 16 else 2
        tonic_votes = defaultdict(int)
        mode_votes = {'Major': 0, 'Minor': 0}

        for i in range(0, len(beat_frames) - step, step):
            start, end = beat_frames[i], beat_frames[i + step]
            segment = self.y[start:end]
            if len(segment) < self.sr * 0.5: continue

            chroma = librosa.feature.chroma_cqt(y=segment, sr=self.sr, bins_per_octave=36)
            avg_chroma = np.mean(chroma, axis=1)
            dominant = int(np.argmax(avg_chroma))
            tonic_votes[dominant] += 1

            if avg_chroma[(dominant + 3) % 12] > avg_chroma[(dominant + 4) % 12]:
                mode_votes['Minor'] += 1
            else:
                mode_votes['Major'] += 1

        if not tonic_votes:
            return {'method': self.name, 'error': 'No votes'}

        most_common = max(tonic_votes, key=tonic_votes.get)
        total_tonic = sum(tonic_votes.values())
        mode = 'Major' if mode_votes['Major'] >= mode_votes['Minor'] else 'Minor'
        confidence = mode_votes[mode] / (mode_votes['Major'] + mode_votes['Minor'] + 1e-10)

        return self.result(most_common, mode, tonic_votes[most_common] / total_tonic, confidence)


# ==================== МЕТОД 6: Квинтовый круг ====================
class CircleOfFifthsMethod(KeyMethod):
    name = "Circle of Fifths"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> Dict:
        avg = self.chroma_avg
        best_score = -np.inf
        best_key, best_mode = 0, 'Major'

        for i in range(12):
            tonic, fifth, fourth = avg[i], avg[(i + 7) % 12], avg[(i + 5) % 12]
            major_score = tonic * 2.0 + fifth * 1.5 + fourth * 1.0
            minor_score = tonic * 2.0 + fifth * 1.5 + avg[(i + 3) % 12] * 1.0

            if major_score > best_score:
                best_score, best_key, best_mode = major_score, i, 'Major'
            if minor_score > best_score:
                best_score, best_key, best_mode = minor_score, i, 'Minor'

        confidence = min(1.0, best_score / (np.max(avg) * 3 + 1e-10))
        return self.result(best_key, best_mode, best_score, confidence)


# ==================== МЕТОД 7: Спектральный анализ ====================
class SpectralMethod(KeyMethod):
    name = "Spectral Analysis"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def detect(self) -> Dict:
        avg = self.chroma_avg
        tonic_idx = int(np.argmax(avg))

        centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        contrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)

        third_diff = avg[(tonic_idx + 4) % 12] - avg[(tonic_idx + 3) % 12]

        # Логика на основе яркости спектра
        if np.mean(centroids) > 1500 and np.mean(contrast[1:]) > 0.5:
            mode = 'Major' if third_diff > 0 else 'Minor'
        else:
            mode = 'Minor' if third_diff < 0 else 'Major'

        confidence = min(1.0, abs(third_diff) * 3 + 0.2)
        return self.result(tonic_idx, mode, avg[tonic_idx], confidence)


# ==================== МЕТОД 8: Essentia ====================
class EssentiaMethod(KeyMethod):
    name = "Essentia"
    weight = METHOD_WEIGHTS.get(name, 1.0)

    def __init__(self, filepath: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath = filepath

    def detect(self) -> Dict:
        if essentia is None or es is None:
            return {'method': self.name, 'error': 'Essentia not installed'}

        try:
            loader = es.MonoLoader(filename=self.filepath)
            audio = loader()
            key_extractor = es.KeyExtractor(profileType='temperley')
            key, scale, strength = key_extractor(audio)

            if not key or not scale:
                return {'method': self.name, 'error': 'Key extraction failed'}

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
    def _autocorrelate_bpm(signal: np.ndarray, sr: int) -> Optional[float]:
        """Вспомогательный метод для поиска BPM через автокорреляцию."""
        if len(signal) < 10: return None
        signal = signal - np.mean(signal)  # Убираем DC offset
        autocorr = np.correlate(signal, signal, mode='full')[len(signal) - 1:]

        min_lag = int(60 * sr / 200)  # 200 BPM
        max_lag = int(60 * sr / 60)  # 60 BPM

        if max_lag >= len(autocorr) or min_lag >= max_lag: return None

        search = autocorr[min_lag:max_lag]
        if len(search) == 0: return None

        peak_lag = np.argmax(search) + min_lag
        bpm = 60 * sr / peak_lag
        return float(bpm) if 30 < bpm < 300 else None

    @staticmethod
    def detect(y: np.ndarray, sr: int) -> Dict[str, float]:
        results = {}

        # 1. Librosa beat tracking
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            results['librosa'] = tempo
        except Exception:
            pass

        # 2. Автокорреляция огибающей
        try:
            envelope = np.mean(np.abs(librosa.stft(y)), axis=0)
            bpm = BPMDetector._autocorrelate_bpm(envelope, sr)
            if bpm: results['autocorrelation'] = bpm
        except Exception:
            pass

        # 3. Автокорреляция спектрального потока
        try:
            spectral_flux = np.diff(np.mean(np.abs(librosa.stft(y)), axis=0))
            spectral_flux = np.maximum(spectral_flux, 0)
            bpm = BPMDetector._autocorrelate_bpm(spectral_flux, sr)
            if bpm: results['spectral_flux'] = bpm
        except Exception:
            pass

        # 4. Автокорреляция onset envelope
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            bpm = BPMDetector._autocorrelate_bpm(onset_env, sr)
            if bpm: results['onset'] = bpm
        except Exception:
            pass

        # Фильтрация выбросов
        if results:
            values = list(results.values())
            median = np.median(values)
            filtered = {k: v for k, v in results.items() if abs(v - median) / median < 0.3}
            return filtered if filtered else results
        return {}


# ==================== ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА ====================
class KeyDetector:
    def __init__(self, filepath: str, verbose: bool = False):
        self.filepath = filepath
        self.verbose = verbose
        self.y, self.sr, self.duration = None, None, 0.0
        self.chroma, self.methods, self.results = None, [], []
        self.bpm_results, self.final_result = {}, {}

        self._load_audio()
        self._prepare_chroma()
        self._init_methods()

    def _load_audio(self):
        logger.info(f"Loading audio: {self.filepath}")
        try:
            y, sr = librosa.load(self.filepath, sr=22050, mono=True)
            y, _ = librosa.effects.trim(y, top_db=20)
            self.y, self.sr = y, sr
            self.duration = len(y) / sr
            logger.info(f"Duration: {self.duration:.1f}s, SR: {sr}Hz")
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def _prepare_chroma(self):
        logger.info("Computing chroma features...")
        self.chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, bins_per_octave=36)

    def _init_methods(self):
        self.methods = [
            ProfileMethod(KRUMHANSL_MAJOR, KRUMHANSL_MINOR, "Krumhansl-Schmuckler", self.y, self.sr, self.chroma),
            ProfileMethod(TEMPERLEY_MAJOR, TEMPERLEY_MINOR, "Temperley", self.y, self.sr, self.chroma),
            BassMethod(self.y, self.sr, self.chroma),
            BoundariesMethod(self.y, self.sr, self.chroma),
            ChordVotingMethod(self.y, self.sr, self.chroma),
            CircleOfFifthsMethod(self.y, self.sr, self.chroma),
            SpectralMethod(self.y, self.sr, self.chroma),
            EssentiaMethod(self.filepath, self.y, self.sr, self.chroma)
        ]

    def run(self) -> Dict:
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
        self.bpm_results = BPMDetector.detect(self.y, self.sr)
        for k, v in self.bpm_results.items():
            logger.info(f"BPM {k}: {v:.1f}")

        self.final_result = self._aggregate_results()
        return self.final_result

    def _aggregate_results(self) -> Dict:
        valid = [r for r in self.results if 'error' not in r and 'pitch_class' in r]
        if not valid:
            return {'error': 'No valid results', 'file': self.filepath}

        groups = defaultdict(lambda: {'votes': 0, 'weighted_sum': 0.0, 'methods': []})

        for res in valid:
            # Используем напрямую pitch_class вместо парсинга строк
            group_key = (res['pitch_class'], res['mode'])
            groups[group_key]['votes'] += 1
            groups[group_key]['weighted_sum'] += res['confidence'] * res.get('weight', 1.0)
            groups[group_key]['methods'].append(res['method'])

        # Сортировка по голосам, затем по взвешенной сумме
        sorted_groups = sorted(groups.items(), key=lambda x: (x[1]['votes'], x[1]['weighted_sum']), reverse=True)

        winner_pc, winner_mode = sorted_groups[0][0]
        winner_data = sorted_groups[0][1]
        total_weighted = sum(g['weighted_sum'] for _, g in groups.items())

        confidence = winner_data['weighted_sum'] / total_weighted if total_weighted > 0 else 0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        level = "ОЧЕНЬ ВЫСОКАЯ" if confidence > 0.8 else "ВЫСОКАЯ" if confidence > 0.6 else "СРЕДНЯЯ" if confidence > 0.4 else "НИЗКАЯ"
        winner_key = pitch_class_to_key(winner_pc, winner_mode)

        bpm_dict = {}
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

        result = {
            'file': self.filepath,
            'duration_seconds': self.duration,
            'sample_rate': self.sr,
            'key': NOTE_NAMES[winner_pc],
            'mode': winner_mode,
            'confidence': confidence,
            'confidence_level': level,
            'votes': winner_data['votes'],
            'total_methods': len(valid),
            'bpm': bpm_dict,
            'all_results': [],
            'voting': {}
        }

        for r in self.results:
            result['all_results'].append({
                'method': r.get('method', 'unknown'),
                'key': r.get('key', 'N/A'),
                'mode': r.get('mode', 'N/A'),
                'confidence': r.get('confidence', 0),
                'error': r.get('error', None)
            })

        for (pc, mode), data in sorted_groups:
            key_str = pitch_class_to_key(pc, mode)
            result['voting'][key_str] = {
                'votes': data['votes'],
                'weighted_confidence': float(data['weighted_sum'] / data['votes']),
                'methods': data['methods']
            }

        return result

    def save_json(self, output_path: Optional[str] = None):
        if not self.final_result:
            logger.warning("No results to save.")
            return

        if output_path is None:
            base = os.path.splitext(self.filepath)[0]
            output_path = base + '_analysis.json'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        res = self.final_result
        if 'error' in res:
            print(f"❌ Error: {res['error']}")
            return

        print("\n" + "=" * 70)
        print(" 🎵 ФИНАЛЬНЫЙ ВЕРДИКТ")
        print("=" * 70)
        print(f"  Тональность: {res['key']} {res['mode']}")
        print(f"  Уверенность: {res['confidence']:.1%} ({res['confidence_level']})")
        print(f"  Согласованность: {res['votes']}/{res['total_methods']} методов")

        if res['bpm']:
            bpm = res['bpm']
            print(f"\n  BPM: среднее {bpm['average']:.1f}, медиана {bpm['median']:.1f}, округлённое {bpm['rounded']}")
            print(f"  Диапазон: {bpm['range'][0]:.1f} – {bpm['range'][1]:.1f}")

        if res['confidence'] < 0.6 and 'Essentia' not in [m['method'] for m in res['all_results'] if
                                                          not m.get('error')]:
            print("\n💡 Рекомендация: установите Essentia для повышения точности (pip install essentia)")

        print("=" * 70 + "\n")


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(description="Определение тональности и BPM аудиофайлов")
    parser.add_argument('files', nargs='+', help='Путь к аудиофайлу(ам)')
    parser.add_argument('--output', '-o', help='Папка для сохранения JSON (по умолчанию рядом с файлами)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять JSON')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    for filepath in args.files:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            continue

        try:
            detector = KeyDetector(filepath, verbose=args.verbose)
            detector.run()
            if not args.no_save:
                out_path = None
                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                    out_name = os.path.splitext(os.path.basename(filepath))[0] + '_analysis.json'
                    out_path = os.path.join(args.output, out_name)
                detector.save_json(out_path)
            detector.print_summary()
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    main()
