import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import defaultdict
import concurrent.futures
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Set, Tuple

import yaml
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency is expected at runtime
    tqdm = None

logger = logging.getLogger("podsponsor")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
PROGRESS_PLAIN_LOG_INTERVAL_SECONDS = 10.0
PROGRESS_EVENT_STATUSES = {
    "queued",
    "transcribing",
    "transcribed",
    "using_cached_llm",
    "analyzing_llm",
    "deriving_ad_blocks",
    "cutting_audio",
    "processed",
    "skipped",
    "failed",
}
ProgressCallback = Callable[[str, Path | None, Dict[str, Any]], None]
FuzzyProgressCallback = Callable[[Dict[str, Any]], None]


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        message = self.format(record)
        if tqdm is not None:
            tqdm.write(message)
        else:  # pragma: no cover
            print(message, file=sys.stderr)


def default_log_file_path(now: datetime | None = None, cwd: Path | None = None) -> Path:
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base_dir = cwd or Path.cwd()
    return base_dir / "logs" / f"podsponsor-{ts}.log"


def configure_logging(use_tqdm_console: bool, log_file: Path | None = None) -> Path:
    log_path = log_file or default_log_file_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)

    if use_tqdm_console and tqdm is not None:
        console_handler: logging.Handler = TqdmLoggingHandler()
    else:
        console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    logger.info("Run log file: %s", log_path)
    return log_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Podsponsor - Podcast Ad Remover")
    parser.add_argument("path", nargs="?", help="Path to podcast MP3 or directory")
    parser.add_argument("--force", action="store_true", help="Reprocess files even if sidecar status is success (ignores backups)")
    parser.add_argument("--update", action="store_true", help="Re-crop audio/SRT from backup based on existing sidecar ad_blocks (skips LLM)")
    parser.add_argument("--dry-run", action="store_true", help="Detect ads and show what would be cut without modifying any files")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--transcribe-only", action="store_true", help="Only generate .srt files and sidecar segments, skip LLM and cutting")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress display mode. auto enables tqdm in TTY and plain progress logs otherwise.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write full run logs to this path (default: ./logs/podsponsor-YYYYMMDD-HHMMSS.log).",
    )
    return parser


def resolve_progress_mode(progress_flag: str, stderr_is_tty: bool, tqdm_available: bool) -> str:
    if progress_flag == "off":
        return "off"
    if progress_flag == "on":
        return "tqdm" if tqdm_available else "plain"
    return "tqdm" if stderr_is_tty and tqdm_available else "plain"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    whole = max(0, int(seconds))
    return f"{whole // 3600:02}:{(whole % 3600) // 60:02}:{whole % 60:02}"


def _truncate_label(label: str, max_len: int = 48) -> str:
    if len(label) <= max_len:
        return label
    return f"...{label[-(max_len - 3):]}"


class RunProgressManager:
    def __init__(self, mode: str):
        self.mode = mode
        self._phase_name = ""
        self._phase_total = 0
        self._phase_done = 0
        self._phase_started_at = 0.0
        self._phase_item_durations: List[float] = []
        self._current_file = "-"
        self._current_status = "-"
        self._last_plain_log = 0.0
        self._last_cross_plain_log = 0.0
        self._phase_bar = None
        self._cross_bar = None

    def start_phase(self, phase_name: str, total: int):
        self.finish_phase()
        self._phase_name = phase_name
        self._phase_total = max(0, total)
        self._phase_done = 0
        self._phase_started_at = time.monotonic()
        self._phase_item_durations = []
        self._current_file = "-"
        self._current_status = "queued"
        self._last_plain_log = 0.0

        if self.mode == "tqdm" and tqdm is not None:
            self._phase_bar = tqdm(
                total=self._phase_total,
                desc=phase_name,
                unit="file",
                dynamic_ncols=True,
            )
            self._refresh_tqdm_postfix()
        elif self.mode == "plain":
            logger.info("Progress: phase=%s total=%d", phase_name, self._phase_total)

    def finish_phase(self):
        if self._phase_bar is not None:
            self._phase_bar.close()
            self._phase_bar = None

    def start_item(self, mp3_path: Path):
        self._current_file = mp3_path.name
        self._current_status = "queued"
        self._refresh_tqdm_postfix()
        self._log_plain_progress(force=True)

    def set_status(self, mp3_path: Path | None, status: str):
        self._current_file = mp3_path.name if mp3_path else self._current_file
        self._current_status = status
        self._refresh_tqdm_postfix()
        self._log_plain_progress(force=True)

    def complete_item(self, duration_seconds: float):
        self._phase_done += 1
        if duration_seconds >= 0:
            self._phase_item_durations.append(duration_seconds)

        if self._phase_bar is not None:
            self._phase_bar.update(1)
        self._refresh_tqdm_postfix()
        self._log_plain_progress(force=True)

    def start_cross_file(self, total_chunks: int):
        if self.mode == "tqdm" and tqdm is not None:
            if self._cross_bar is not None:
                self._cross_bar.close()
            self._cross_bar = tqdm(
                total=max(0, total_chunks),
                desc="Cross-file matching",
                unit="chunk",
                dynamic_ncols=True,
            )
        elif self.mode == "plain":
            logger.info("Progress: cross-file chunks total=%d", total_chunks)
            self._last_cross_plain_log = 0.0

    def update_cross_file(self, chunks_done: int, chunks_total: int, comparisons: int, matches: int, elapsed: float):
        if self._cross_bar is not None:
            self._cross_bar.total = max(0, chunks_total)
            self._cross_bar.n = max(0, chunks_done)
            self._cross_bar.set_postfix({"cmp": comparisons, "matches": matches, "elapsed": f"{elapsed:.0f}s"}, refresh=False)
            self._cross_bar.refresh()
            if chunks_done >= chunks_total:
                self._cross_bar.close()
                self._cross_bar = None
        elif self.mode == "plain":
            now = time.monotonic()
            if (now - self._last_cross_plain_log) < PROGRESS_PLAIN_LOG_INTERVAL_SECONDS and chunks_done < chunks_total:
                return
            self._last_cross_plain_log = now
            logger.info(
                "Progress: cross-file %d/%d chunks comparisons=%d matches=%d elapsed=%ss",
                chunks_done,
                chunks_total,
                comparisons,
                matches,
                int(elapsed),
            )

    def _refresh_tqdm_postfix(self):
        if self._phase_bar is None:
            return
        left = max(0, self._phase_total - self._phase_done)
        eta_seconds: float | None = None
        if self._phase_item_durations and left > 0:
            eta_seconds = (sum(self._phase_item_durations) / len(self._phase_item_durations)) * left
        self._phase_bar.set_postfix(
            {
                "left": left,
                "eta": _format_duration(eta_seconds),
                "current": _truncate_label(self._current_file),
                "status": self._current_status,
            },
            refresh=False,
        )
        self._phase_bar.refresh()

    def _log_plain_progress(self, force: bool = False):
        if self.mode != "plain":
            return
        now = time.monotonic()
        if not force and (now - self._last_plain_log) < PROGRESS_PLAIN_LOG_INTERVAL_SECONDS:
            return
        self._last_plain_log = now

        left = max(0, self._phase_total - self._phase_done)
        elapsed = now - self._phase_started_at if self._phase_started_at else 0.0
        eta_seconds: float | None = None
        if self._phase_item_durations and left > 0:
            eta_seconds = (sum(self._phase_item_durations) / len(self._phase_item_durations)) * left

        logger.info(
            "Progress: phase=%s done=%d/%d left=%d elapsed=%s eta=%s current=%s status=%s",
            self._phase_name,
            self._phase_done,
            self._phase_total,
            left,
            _format_duration(elapsed),
            _format_duration(eta_seconds),
            self._current_file,
            self._current_status,
        )

    def close(self):
        self.finish_phase()
        if self._cross_bar is not None:
            self._cross_bar.close()
            self._cross_bar = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


class PodsponsorConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

        # Whisper settings
        self.w_model = self.data.get("whisper", {}).get("model", "medium")
        self.w_device = self.data.get("whisper", {}).get("device", "cuda")
        self.w_compute = self.data.get("whisper", {}).get("compute_type", "float16")
        self.w_batch_size = int(self.data.get("whisper", {}).get("batch_size", 16))
        self.w_chunk_size = int(self.data.get("whisper", {}).get("chunk_size", 20))

        # LLM settings
        llm = self.data.get("llm", {})
        self.summary_lang = llm.get("summary_language", "en")

        default_prompt = (
            "You are an expert podcast analyst.\n"
            "You must do two things:\n"
            "1) Identify sponsor/ad-read blocks in the transcript.\n"
            "2) Write a strict, useful episode summary.\n\n"
            "Language rules:\n"
            "- The entire episode summary must be written strictly in {summary_language} (including headings).\n"
            "- Keep emojis and heading levels exactly as specified below.\n\n"
            "Summary rules (very important):\n"
            "- The summary must NOT mention sponsors, ads, discount codes, promo segments, host promos, calls-to-action, or housekeeping.\n"
            "- Exclude intros/outros and any ad-read content from the summary.\n"
            "- Do not include timestamps, timecodes, or segment indices like [12].\n"
            "- Only include factual information present in the transcript. If unsure, omit it.\n"
            "- No filler, no speculation, no commentary about being an AI, no meta about the transcript.\n\n"
            "The `summary` field MUST be valid Markdown and MUST follow this exact structure.\n"
            "Translate the words after the emoji into {summary_language}, but keep the emojis and heading levels the same.\n"
            "Required sections: Title, Overview, Topics (>= 1 topic), Key Takeaways.\n"
            "Optional sections: References Mentioned. Omit optional sections entirely if nothing is mentioned.\n"
            "\n"
            "Structure:\n"
            "# 🎧 <Episode Summary Title>\n"
            "## 🧭 <Overview>\n"
            "- <1-4 bullets: what this episode is about>\n"
            "## 🧩 <Topics>\n"
            "### <Topic 1>\n"
            "- <2-5 bullets: concrete points, claims, examples>\n"
            "### <Topic 2>\n"
            "- <2-5 bullets>\n"
            "## 🧠 <Key Takeaways>\n"
            "- <3-8 bullets: actionable ideas, mental models>\n"
            "## 📚 <References Mentioned>\n"
            "- 📖 <Book Title> — <Author if mentioned>\n"
            "- 🎬 <Film/Series Title>\n"
            "- 👤 <Person Name> — <Why they matter in context>\n"
            "- 🛠️ <Tool/Project Name> — <What it is used for>\n"
            "- 🔗 <Other resource worth looking up>\n"
            "\n"
            "Transcript format:\n"
            "- You will receive transcribed audio segments. Each segment has an ID like [0].\n"
            "- Some segments are marked with [REPEATED]. These are strong hints for pre-recorded ads/promos/intro/outro.\n\n"
            "- Some segments are marked with [HIGH_FREQ]. This means segment frequency >= 3 across episodes.\n\n"
            "Ad detection instructions:\n"
            "1) Treat [REPEATED] segments as high-signal ad/promos.\n"
            "2) Mark a segment as ad only when more than 50% of that segment is ad/promo content.\n"
            "3) If only a small part of a segment is ad and most of it is regular podcast content, do NOT mark that segment as ad.\n"
            "4) Even if a segment does not look ad-like on wording alone, if it is long enough (about 2-3 sentences) and marked [HIGH_FREQ], increase ad likelihood.\n"
            "5) Expand slightly before/after [REPEATED] to capture the full natural boundary of the ad read.\n"
            "6) Look for transitions like \"This episode is brought to you by...\", promo codes, URLs, or product pitches.\n"
            "7) Group contiguous ad segments into a single block.\n"
            "Return detected ad blocks with their starting and ending segment indices (and confidence)."
        )
        self.prompt_template = llm.get("prompt", default_prompt)
        
        self.llm_providers = llm.get("providers", [])
        if not self.llm_providers:
            # Fallback if config is outdated
            self.llm_providers = [{
                "base_url": llm.get("base_url", "http://localhost:11434/v1"),
                "model": llm.get("model", "llama3"),
                "api_key": llm.get("api_key", "dummy"),
                "timeout_seconds": float(llm.get("timeout_seconds", 600)),
            }]

        # Detection settings
        detection = self.data.get("detection", {})
        self.sim_thresh = float(detection.get("similarity_threshold", 0.80))
        self.silence_db = int(detection.get("silence_db", -40))
        self.silence_min_dur = float(detection.get("silence_min_duration", 1.5))
        self.min_confidence = float(detection.get("min_confidence", 0.70))
        self.min_ad_duration = float(detection.get("min_ad_duration", 8.0))

        # Backup settings
        backup = self.data.get("backup", {})
        self.backup_enabled = bool(backup.get("enabled", True))
        backup_location = backup.get("location", "backup")
        self.backup_location = str(backup_location or "backup").strip() or "backup"

SIDECAR_VERSION = 2


def sidecar_path_for_mp3(mp3_path: Path) -> Path:
    return mp3_path.with_name(f"{mp3_path.stem}.podsponsor.json")


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _to_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

def default_sidecar(config: PodsponsorConfig) -> dict:
    return {
        "version": SIDECAR_VERSION,
        "status": "new",
        "processing_info": {
            "processed_at": None,
            "processing_time_seconds": None,
            "transcription_language": None,
            "whisper_model": str(config.w_model),
        },
        "backup_path": None,
        "backup_srt_path": None,
        "ad_blocks": [],
        "original_segments": [],
    }


def normalize_sidecar(sidecar: dict | None, config: PodsponsorConfig) -> dict:
    if not isinstance(sidecar, dict):
        sidecar = {}

    normalized = default_sidecar(config)
    normalized["version"] = _to_int(sidecar.get("version", SIDECAR_VERSION), SIDECAR_VERSION)

    status = sidecar.get("status")
    if status in {"new", "transcripted", "success"}:
        normalized["status"] = status

    processing_info = sidecar.get("processing_info")
    if isinstance(processing_info, dict):
        normalized["processing_info"].update(
            {
                "processed_at": processing_info.get("processed_at"),
                "processing_time_seconds": processing_info.get("processing_time_seconds"),
                "transcription_language": processing_info.get("transcription_language"),
                "whisper_model": processing_info.get("whisper_model") or normalized["processing_info"]["whisper_model"],
            }
        )

    backup_path = sidecar.get("backup_path")
    if isinstance(backup_path, str):
        normalized["backup_path"] = backup_path

    backup_srt_path = sidecar.get("backup_srt_path")
    if isinstance(backup_srt_path, str):
        normalized["backup_srt_path"] = backup_srt_path

    ad_blocks = sidecar.get("ad_blocks")
    if isinstance(ad_blocks, list):
        normalized_blocks: List[dict] = []
        for block in ad_blocks:
            if not isinstance(block, dict):
                continue
            normalized_blocks.append(
                {
                    "start": _to_float(block.get("start")),
                    "end": _to_float(block.get("end")),
                    "text": str(block.get("text", "")),
                    "confidence": _to_float(block.get("confidence")),
                    "frequency": _to_int(block.get("frequency")),
                    "source": str(block.get("source", "llm")),
                }
            )
        normalized["ad_blocks"] = normalized_blocks

    # Support old "segments" key or new "original_segments"
    segments = sidecar.get("original_segments") or sidecar.get("segments")
    if isinstance(segments, list):
        normalized_segments: List[dict] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            normalized_segments.append(
                {
                    "start": _to_float(seg.get("start")),
                    "end": _to_float(seg.get("end")),
                    "text": str(seg.get("text", "")),
                    "frequency": _to_int(seg.get("frequency")),
                }
            )
        normalized["original_segments"] = normalized_segments

    return normalized


def check_silence(input_path: Path, threshold_db: int = -40, duration_sec: float = 1.5) -> List[Tuple[float, float]]:
    """Detect silence intervals using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=n={threshold_db}dB:d={duration_sec}",
        "-f",
        "null",
        "-",
    ]

    intervals: List[Tuple[float, float]] = []
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        start = None
        for line in res.stderr.splitlines():
            if "silence_start" in line:
                parts = line.split("silence_start: ")
                if len(parts) > 1:
                    start = float(parts[1])
            elif "silence_end" in line and start is not None:
                parts = line.split("silence_end: ")
                if len(parts) > 1:
                    end_part = parts[1].split(" |")[0]
                    end = float(end_part)
                    intervals.append((start, end))
                    start = None
    except subprocess.CalledProcessError as exc:
        logger.error("Silence detection failed: %s", exc.stderr)

    return intervals


# --- Transcriber ---
class Transcriber:
    def __init__(self, config: PodsponsorConfig):
        self.config = config
        self._model = None
        
        # Suppress aggressive Pyannote warnings about TF32
        try:
            import torch
            import warnings

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        except ImportError:
            pass

    def transcribe(self, audio_path: Path, language: str | None = None) -> dict:
        import whisperx

        if self._model is None:
            logger.info("Loading WhisperX model '%s' on %s (language=%s)", self.config.w_model, self.config.w_device, language or 'auto')
            self._model = whisperx.load_model(
                self.config.w_model,
                self.config.w_device,
                compute_type=self.config.w_compute,
                language=language,
            )

        audio = whisperx.load_audio(str(audio_path))
        logger.info("Transcribing %s...", audio_path.name)
        try:
            result = self._model.transcribe(
                audio,
                batch_size=self.config.w_batch_size,
                language=language,
                chunk_size=self.config.w_chunk_size,
            )
        except TypeError:
            logger.warning(
                "WhisperX runtime does not accept chunk_size; using legacy transcribe call without chunk_size."
            )
            result = self._model.transcribe(audio, batch_size=self.config.w_batch_size, language=language)
        return result


def format_srt_ts(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    whole = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    return f"{whole // 3600:02}:{(whole % 3600) // 60:02}:{whole % 60:02},{ms:03}"


def parse_srt_ts(value: str) -> float:
    hhmmss, ms = value.strip().split(",")
    hours, minutes, seconds = hhmmss.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(ms) / 1000.0)


def save_srt(segments: List[dict], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_ts(segment['start'])} --> {format_srt_ts(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")

# --- Cross-File Matching ---

# Global state for fast forked multiprocessing (zero-copy overhead on Linux).
# Keys: unique_texts (List[str]), text_to_file_indices (Dict[str, List[Tuple[Path, int]]]),
#        text_trigrams (List[Set[str]]), text_lens (List[int]),
#        sim_threshold (float), is_new (List[bool])
_FUZZY_STATE: Dict[str, object] = {}

def _process_fuzzy_chunk(start_idx: int, end_idx: int) -> Tuple[int, int, List[Tuple[Path, int]]]:
    unique_texts = _FUZZY_STATE["unique_texts"]
    text_to_file_indices = _FUZZY_STATE["text_to_file_indices"]
    text_trigrams = _FUZZY_STATE["text_trigrams"]
    text_lens = _FUZZY_STATE["text_lens"]
    sim_threshold = _FUZZY_STATE["sim_threshold"]
    is_new = _FUZZY_STATE["is_new"]

    comparisons = 0
    matches = 0
    suspicious_pairs = []
    
    n = len(unique_texts)
    
    for i in range(start_idx, end_idx):
        is_new_b = is_new[i]
        text_b = unique_texts[i]
        files_b = {fp for fp, _ in text_to_file_indices[text_b]}
        len_b = text_lens[i]
        tri_b = text_trigrams[i]
        
        sm = SequenceMatcher(None, "", text_b)
        
        for j in range(i + 1, n):
            # 1-to-N OPTIMIZATION: Skip comparing two old texts
            if not is_new_b and not is_new[j]:
                continue

            len_a = text_lens[j]
            if len_a - len_b > len_b * 0.3:
                break
                
            text_a = unique_texts[j]
            files_a = {fp for fp, _ in text_to_file_indices[text_a]}
            if not (files_a - files_b):
                continue
                
            tri_a = text_trigrams[j]
            overlap = len(tri_a & tri_b)
            min_tri = min(len(tri_a), len(tri_b))
            if min_tri > 0 and (overlap / min_tri) < 0.3:
                continue
                
            sm.set_seq1(text_a)
            comparisons += 1
            if sm.ratio() > sim_threshold:
                matches += 1
                suspicious_pairs.extend(text_to_file_indices[text_b])
                suspicious_pairs.extend(text_to_file_indices[text_a])
                
    return comparisons, matches, suspicious_pairs


def find_repeated_segments(
    targets: List[Path],
    new_targets: Set[Path],
    sim_threshold: float,
    load_all_segments_func=None,
    fuzzy_progress_callback: FuzzyProgressCallback | None = None,
) -> Dict[Path, Set[int]]:
    """Find segments that appear in multiple files (likely ads).

    Uses exact-match hashing on normalized text for O(n) performance.
    Also runs fuzzy matching via SequenceMatcher for near-duplicates.
    Optimized for 1-to-N matching: only processes when new targets are present.
    """
    import time

    t0 = time.time()
    logger.info("Starting cross-file match for %d total targets (%d new targets to compare)", len(targets), len(new_targets))

    if not new_targets:
        logger.info("No new targets to cross-match. Returning empty results.")
        return {}

    if load_all_segments_func is None:
        raise ValueError("load_all_segments_func must be provided when calling find_repeated_segments")
    all_segments = load_all_segments_func(targets)

    total_segs = sum(len(s) for s in all_segments.values())
    logger.info("Cross-file matching: starting matching on %d files, %d total segments (threshold=%.2f)",
                len(targets), total_segs, sim_threshold)

    # --- Phase 1: Build Index ---
    text_to_files: Dict[str, Set[Path]] = defaultdict(set)
    text_to_locations: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)

    for file_path, segments in all_segments.items():
        for idx, seg in enumerate(segments):
            normalized = seg.get("text", "").strip().lower()
            if len(normalized) < 20:
                continue
            text_to_files[normalized].add(file_path)
            text_to_locations[normalized].append((file_path, idx))

    t1 = time.time()
    logger.info("Cross-file matching: indexing done in %.1fs — %d unique texts indexed", t1 - t0, len(text_to_files))

    # --- Phase 2: Exact Matches ---
    suspicious: Dict[Path, Set[int]] = defaultdict(set)
    for normalized, files in text_to_files.items():
        if len(files) >= 2:
            # 1-to-N OPTIMIZATION: Only flag if at least one file is in new_targets
            if any(f in new_targets for f in files):
                for file_path, idx in text_to_locations[normalized]:
                    suspicious[file_path].add(idx)

    exact_count = sum(len(v) for v in suspicious.values())
    t2 = time.time()
    logger.info("Cross-file matching: exact-match phase done in %.1fs — %d exact duplicates across %d files",
                t2 - t1, exact_count, len(suspicious))

    # --- Phase 3: Fuzzy Matching (with SM set_seq2 caching + global text dedup) ---
    # Collect unique texts not already flagged, with their (file, idx) mappings
    text_to_file_indices: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for file_path, segments in all_segments.items():
        for idx, seg in enumerate(segments):
            normalized = seg.get("text", "").strip().lower()
            if len(normalized) < 20:
                continue
            if idx not in suspicious.get(file_path, set()):
                text_to_file_indices[normalized].append((file_path, idx))

    # Deduplicate: flatten to unique text list with file mappings
    unique_texts = list(text_to_file_indices.keys())
    # Pre-sort by length for efficient length-filter skipping
    unique_texts.sort(key=len)

    # 1-to-N OPTIMIZATION: Pre-calculate is_new array for fast access in fuzzy loop
    is_new = [
        any(fp in new_targets for fp, _ in text_to_file_indices[t])
        for t in unique_texts
    ]

    n = len(unique_texts)
    logger.info("Cross-file matching: fuzzy phase starting — %d unique texts to compare (from %d file-segments)",
                n, sum(len(v) for v in text_to_file_indices.values()))

    # Pre-compute trigrams for fast LSH-style filtering
    text_trigrams = [set(t[k:k+3] for k in range(max(1, len(t)-2))) for t in unique_texts]
    text_lens = [len(t) for t in unique_texts]

    fuzzy_comparisons = 0
    fuzzy_matches = 0
    t3 = time.time()
    last_log = t3

    # Populate global state for fork-based ProcessPoolExecutor
    global _FUZZY_STATE
    _FUZZY_STATE = {
        "unique_texts": unique_texts,
        "text_to_file_indices": text_to_file_indices,
        "text_trigrams": text_trigrams,
        "text_lens": text_lens,
        "sim_threshold": sim_threshold,
        "is_new": is_new,
    }

    # Decide chunks based on CPU cores
    num_cores = os.cpu_count() or 4
    num_chunks = max(1, num_cores * 4)
    chunk_size = max(1, n // num_chunks)
    chunks = []
    
    # Calculate boundaries
    idx = 0
    while idx < n:
        chunks.append((idx, min(idx + chunk_size, n)))
        idx += chunk_size

    if fuzzy_progress_callback is not None:
        fuzzy_progress_callback(
            {
                "chunks_done": 0,
                "chunks_total": len(chunks),
                "comparisons": fuzzy_comparisons,
                "matches": fuzzy_matches,
                "elapsed": 0.0,
            }
        )

    def _update_fuzzy_progress(chunks_done: int):
        nonlocal last_log, fuzzy_comparisons, fuzzy_matches
        now = time.time()
        if fuzzy_progress_callback is not None:
            fuzzy_progress_callback(
                {
                    "chunks_done": chunks_done,
                    "chunks_total": len(chunks),
                    "comparisons": fuzzy_comparisons,
                    "matches": fuzzy_matches,
                    "elapsed": now - t3,
                }
            )
        if now - last_log >= 30 or chunks_done == len(chunks):
            elapsed = now - t3
            logger.info(
                "Cross-file matching: fuzzy progress %d/%d chunks (%.0f%%) — %d comparisons, %d matches, %.0fs elapsed",
                chunks_done, len(chunks), 100 * chunks_done / len(chunks),
                fuzzy_comparisons, fuzzy_matches, elapsed
            )
            last_log = now

    fork_ctx = None
    try:
        fork_ctx = multiprocessing.get_context("fork")
    except (ValueError, AttributeError):
        logger.warning(
            "Cross-file matching: 'fork' multiprocessing context is unavailable on this platform; "
            "falling back to in-process fuzzy matching (slower)."
        )

    if fork_ctx is None:
        logger.info("Cross-file matching: running in-process across %d chunks...", len(chunks))
        chunks_done = 0
        for start_i, end_i in chunks:
            comps, mats, susp_pairs = _process_fuzzy_chunk(start_i, end_i)
            fuzzy_comparisons += comps
            fuzzy_matches += mats
            for fp, susp_idx in susp_pairs:
                suspicious[fp].add(susp_idx)
            chunks_done += 1
            _update_fuzzy_progress(chunks_done)
    else:
        logger.info("Cross-file matching: spawning %d processes across %d chunks...", num_cores, len(chunks))
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores, mp_context=fork_ctx) as executor:
            futures = {executor.submit(_process_fuzzy_chunk, start_i, end_i): (start_i, end_i) for start_i, end_i in chunks}

            chunks_done = 0
            for future in concurrent.futures.as_completed(futures):
                comps, mats, susp_pairs = future.result()
                fuzzy_comparisons += comps
                fuzzy_matches += mats
                for fp, susp_idx in susp_pairs:
                    suspicious[fp].add(susp_idx)

                chunks_done += 1
                _update_fuzzy_progress(chunks_done)

    t4 = time.time()
    total = sum(len(v) for v in suspicious.values())
    logger.info(
        "Cross-file matching: DONE in %.1fs total (index=%.1fs, exact=%.1fs, fuzzy=%.1fs) — "
        "%d suspicious segments across %d files, %d fuzzy comparisons, %d fuzzy matches",
        t4 - t0, t1 - t0, t2 - t1, t4 - t3,
        total, len(suspicious), fuzzy_comparisons, fuzzy_matches
    )

    return dict(suspicious)


# --- LLM Client ---
def analyze_with_llm(
    config: PodsponsorConfig,
    segments: List[dict],
    suspicious_indices: Set[int],
) -> Tuple[dict, dict]:
    from openai import OpenAI

    transcript_lines = []
    for i, segment in enumerate(segments):
        text = segment["text"].strip()
        labels: List[str] = []
        if i in suspicious_indices:
            labels.append("[REPEATED]")
        if _to_int(segment.get("frequency", 0)) >= 3:
            labels.append("[HIGH_FREQ]")
        label_prefix = f"{' '.join(labels)} " if labels else ""
        transcript_lines.append(f"[{i}] {label_prefix}{text}")

    transcript_text = "\n".join(transcript_lines)
    prompt = config.prompt_template.format(summary_language=config.summary_lang)
    
    # Define Structured Output Schema
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ad_detection_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ads": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_index": {"type": "integer"},
                                "end_index": {"type": "integer"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["start_index", "end_index", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "summary": {"type": "string"}
                },
                "required": ["ads", "summary"],
                "additionalProperties": False
            }
        }
    }

    last_error = None
    for provider in config.llm_providers:
        client = OpenAI(
            base_url=provider.get("base_url"),
            api_key=provider.get("api_key", "dummy"),
            timeout=provider.get("timeout_seconds", 600.0),
            max_retries=provider.get("max_retries", 2)
        )
        model = provider.get("model")
        logger.info("Calling LLM API using provider: %s (model: %s)", provider.get("base_url"), model)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript_text},
                ],
                temperature=0.0,
                response_format=schema,
            )

            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty response from LLM")

            parsed = json.loads(response_content)
            usage_raw = getattr(response, "usage", None)
            usage = {
                "prompt_tokens": _to_int(getattr(usage_raw, "prompt_tokens", 0)),
                "completion_tokens": _to_int(getattr(usage_raw, "completion_tokens", 0)),
                "total_tokens": _to_int(getattr(usage_raw, "total_tokens", 0)),
            }
            llm_response_payload = {
                "provider": provider.get("base_url"),
                "model": model,
                "usage": usage,
                "choices": [
                    {
                        "text": response_content,
                        "index": _to_int(getattr(response.choices[0], "index", 0)),
                        "logprobs": getattr(response.choices[0], "logprobs", None),
                        "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    }
                ],
                "parsed": parsed,
            }

            return parsed, llm_response_payload
            
        except Exception as exc:
            logger.warning("LLM provider %s failed: %s", provider.get("base_url"), exc)
            last_error = exc

    logger.error("All LLM providers failed.")
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def parse_llm_ad_blocks(raw_ads, max_index: int, min_confidence: float) -> List[Tuple[int, int, float]]:
    """Parse LLM ad blocks and filter by confidence. Returns list of (start, end, confidence)."""
    blocks: List[Tuple[int, int, float]] = []
    if not isinstance(raw_ads, list):
        return blocks

    for item in raw_ads:
        if not isinstance(item, dict):
            continue

        try:
            start = int(item.get("start_index", -1))
            end = int(item.get("end_index", -1))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        if start < 0 or end < 0 or start > end or end >= max_index:
            continue

        if confidence < min_confidence:
            logger.info("Dropping low-confidence ad block [%d-%d] (%.2f < %.2f)", start, end, confidence, min_confidence)
            continue

        blocks.append((start, end, confidence))

    return blocks


def group_contiguous(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []

    blocks: List[List[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            blocks.append(current)
            current = [idx]
    blocks.append(current)
    return blocks


def normalize_cut_regions(cut_regions: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    clamped: List[Tuple[float, float]] = []
    for start, end in cut_regions:
        s = max(0.0, min(total_duration, float(start)))
        e = max(0.0, min(total_duration, float(end)))
        if e <= s:
            continue
        clamped.append((s, e))

    if not clamped:
        return []

    clamped.sort(key=lambda item: item[0])
    merged: List[Tuple[float, float]] = [clamped[0]]

    for start, end in clamped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-6:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def compute_keep_regions(cut_regions: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    keep_regions: List[Tuple[float, float]] = []
    last_end = 0.0

    for start, end in cut_regions:
        if start > last_end:
            keep_regions.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < total_duration:
        keep_regions.append((last_end, total_duration))

    return [(start, end) for start, end in keep_regions if (end - start) > 0.02]


def get_time_mapping(t: float, keep_regions: List[Tuple[float, float]]) -> float | None:
    """Map old time `t` to new time, returning None if `t` falls in a cut region."""
    new_t = 0.0
    for k_start, k_end in keep_regions:
        if t < k_start:
            return None
        if t <= k_end:
            return new_t + (t - k_start)
        new_t += (k_end - k_start)
    return None


def shift_transcript(segments: List[dict], keep_regions: List[Tuple[float, float]]) -> List[dict]:
    """Shift timestamps of segments mathematically based on what regions were kept."""
    shifted = []
    
    for seg in segments:
        new_words = []
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
                
            w_start_new = get_time_mapping(w["start"], keep_regions)
            w_end_new = get_time_mapping(w["end"], keep_regions)
            
            # Require both bounds to be preserved inside a kept region
            if w_start_new is not None and w_end_new is not None:
                new_w = dict(w)
                new_w["start"] = w_start_new
                new_w["end"] = w_end_new
                new_words.append(new_w)
                
        if seg.get("words"):
            if not new_words:
                continue # Entire segment's words were cut
                
            new_seg = dict(seg)
            new_seg["words"] = new_words
            new_seg["start"] = new_words[0]["start"]
            new_seg["end"] = new_words[-1]["end"]
            
            if len(new_words) < len(seg["words"]):
                # Rebuild text if words were removed
                new_seg["text"] = "".join(w.get("word", "") for w in new_words)
                
            shifted.append(new_seg)
        else:
            # Segment without word-level timings
            s_start = get_time_mapping(seg["start"], keep_regions)
            s_end = get_time_mapping(seg["end"], keep_regions)
            
            if s_start is not None and s_end is not None:
                new_seg = dict(seg)
                new_seg["start"] = s_start
                new_seg["end"] = s_end
                shifted.append(new_seg)

    return shifted


def get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Failed to get audio duration for {path.name}: {exc}") from exc


# --- Processor Engine ---
class Processor:
    def __init__(self, config_path: str, progress_callback: ProgressCallback | None = None):
        self.config = PodsponsorConfig(config_path)
        self.transcriber = Transcriber(self.config)
        self._show_languages: Dict[Path, str | None] = {}
        self._sidecar_cache: Dict[Path, dict] = {}
        self._segments_mem_cache: Dict[Path, List[dict]] = {}
        self._progress_callback = progress_callback

    def _emit_progress(self, event: str, mp3_path: Path | None, **fields):
        if event not in PROGRESS_EVENT_STATUSES:
            return
        if self._progress_callback is None:
            return
        try:
            payload = dict(fields)
            payload["event"] = event
            self._progress_callback(event, mp3_path, payload)
        except Exception as exc:  # pragma: no cover - defensive callback isolation
            logger.debug("Progress callback failed: %s", exc)

    def _get_show_language(self, show_dir: Path) -> str | None:
        """Read language from metadata.json if present."""
        if show_dir not in self._show_languages:
            meta_path = show_dir / "metadata.json"
            lang = None
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    lang = meta.get("language")
                    if lang:
                        logger.info("Using language '%s' from %s", lang, meta_path)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read metadata.json: %s", exc)
            self._show_languages[show_dir] = lang
        return self._show_languages[show_dir]

    def _load_sidecar(self, mp3_path: Path) -> dict:
        if mp3_path in self._sidecar_cache:
            return self._sidecar_cache[mp3_path]

        sidecar_path = sidecar_path_for_mp3(mp3_path)
        raw: dict | None = None
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not load sidecar %s: %s", sidecar_path.name, exc)

        sidecar = normalize_sidecar(raw, self.config)
        self._sidecar_cache[mp3_path] = sidecar

        if not sidecar_path.exists():
            save_json_atomic(sidecar_path, sidecar)

        return sidecar

    def _save_sidecar(self, mp3_path: Path):
        save_json_atomic(sidecar_path_for_mp3(mp3_path), self._sidecar_cache[mp3_path])

    def _ensure_sidecar(self, mp3_path: Path) -> dict:
        return self._load_sidecar(mp3_path)

    def _segments_for_storage(self, segments: List[dict]) -> List[dict]:
        return [
            {
                "start": _to_float(seg.get("start")),
                "end": _to_float(seg.get("end")),
                "text": str(seg.get("text", "")),
                "frequency": 0,
            }
            for seg in segments
        ]

    def ensure_transcription(self, mp3_path: Path):
        """Phase 1: Ensure sidecar segments and .srt exist for this file."""
        self._emit_progress("transcribing", mp3_path, phase="transcription")
        sidecar = self._ensure_sidecar(mp3_path)
        status = sidecar.get("status")
        if status in {"transcripted", "success"} and (sidecar.get("original_segments") or sidecar.get("segments")):
            logger.info("Segments exist in sidecar, skipping transcription: %s", mp3_path.name)
            self._emit_progress("skipped", mp3_path, phase="transcription", reason="already_transcribed")
            return

        started_at = time.monotonic()
        srt_path = mp3_path.with_suffix(".srt")
        language = self._get_show_language(mp3_path.parent)
        logger.info("Transcribing (full run): %s", mp3_path.name)
        try:
            result = self.transcriber.transcribe(mp3_path, language=language)
        except Exception:
            self._emit_progress("failed", mp3_path, phase="transcription", reason="transcribe_failed")
            raise
        segments = self._segments_for_storage(result.get("segments", []))

        if not segments:
            logger.warning("No segments produced for %s", mp3_path.name)
            self._emit_progress("failed", mp3_path, phase="transcription", reason="no_segments")
            return

        sidecar["original_segments"] = segments
        sidecar["status"] = "transcripted"
        sidecar["ad_blocks"] = []
        info = sidecar.setdefault("processing_info", {})
        info["transcription_language"] = result.get("language") or language
        info["whisper_model"] = str(self.config.w_model)
        info["processed_at"] = None
        info["processing_time_seconds"] = None
        self._save_sidecar(mp3_path)

        save_srt(segments, srt_path)
        logger.info("Saved sidecar segments and SRT for: %s", mp3_path.name)
        self._emit_progress(
            "transcribed",
            mp3_path,
            phase="transcription",
            duration_seconds=time.monotonic() - started_at,
            segments=len(segments),
        )

    def update_segment_frequencies(self, targets: List[Path]):
        """Fill segment frequency using sidecars in transcripted/success statuses."""
        text_to_files: Dict[str, Set[Path]] = defaultdict(set)
        sidecars: Dict[Path, dict] = {}

        for mp3_path in targets:
            sidecar = self._ensure_sidecar(mp3_path)
            sidecars[mp3_path] = sidecar
            if sidecar.get("status") not in {"transcripted", "success"}:
                continue
            for seg in sidecar.get("original_segments", []):
                normalized = seg.get("text", "").strip().lower()
                if len(normalized) < 20:
                    continue
                text_to_files[normalized].add(mp3_path)

        for mp3_path in targets:
            sidecar = sidecars[mp3_path]
            if sidecar.get("status") != "transcripted":
                continue

            changed = False
            for seg in sidecar.get("original_segments", []):
                normalized = seg.get("text", "").strip().lower()
                freq = len(text_to_files.get(normalized, set())) if len(normalized) >= 20 else 0
                if seg.get("frequency") != freq:
                    seg["frequency"] = freq
                    changed = True

            if changed:
                self._save_sidecar(mp3_path)

    def preload_all_segments(self, targets: List[Path]):
        """Preload all sidecar segments into memory sequentially."""
        logger.info("Pre-loading segments into memory sequentially...")
        for mp3_path in targets:
            sidecar = self._ensure_sidecar(mp3_path)
            original_segments = sidecar.get("original_segments", [])
            if original_segments:
                self._segments_mem_cache[mp3_path] = original_segments
            
        logger.info("Pre-loaded %d segment files into memory.", len(self._segments_mem_cache))

    def load_all_segments(self, targets: List[Path]) -> Dict[Path, List[dict]]:
        """Return segments for all target files from memory cache."""
        all_segments: Dict[Path, List[dict]] = {}
        for mp3_path in targets:
            if mp3_path in self._segments_mem_cache:
                all_segments[mp3_path] = self._segments_mem_cache[mp3_path]
        return all_segments

    def _save_summary(self, mp3_path: Path, summary: str):
        md_path = mp3_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info("Saved Summary: %s", md_path.name)

    def _derive_ad_blocks(
        self,
        mp3_path: Path,
        segments: List[dict],
        parsed_blocks: List[Tuple[int, int, float]],
    ) -> List[dict]:
        if not parsed_blocks:
            return []

        confidence_by_index: Dict[int, float] = {}
        all_ad_indices: set[int] = set()
        for start, end, confidence in parsed_blocks:
            for idx in range(start, end + 1):
                all_ad_indices.add(idx)
                confidence_by_index[idx] = max(confidence_by_index.get(idx, 0.0), confidence)

        silences = check_silence(mp3_path, self.config.silence_db, self.config.silence_min_dur)
        blocks = group_contiguous(sorted(all_ad_indices))
        final_ad_blocks: List[dict] = []

        for block in blocks:
            start_seg = segments[block[0]]
            end_seg = segments[block[-1]]

            seg_start = _to_float(start_seg.get("start"))
            seg_end = _to_float(end_seg.get("end"))

            if (seg_end - seg_start) < self.config.min_ad_duration:
                logger.info(
                    "Dropping short ad block (%.1fs < %.1fs): segments %d-%d",
                    seg_end - seg_start,
                    self.config.min_ad_duration,
                    block[0],
                    block[-1],
                )
                continue

            final_start = seg_start
            final_end = seg_end

            for silence_start, silence_end in silences:
                if silence_end < final_start and (final_start - silence_end) < 15.0:
                    final_start = silence_end

            for silence_start, silence_end in silences:
                if silence_start > final_end:
                    if (silence_start - final_end) < 15.0:
                        final_end = silence_start
                    break

            ad_text = " ".join(segments[i].get("text", "") for i in block).strip()
            block_confidence = max(confidence_by_index.get(i, 0.0) for i in block)
            block_frequency = min(_to_int(segments[i].get("frequency", 0)) for i in block)

            logger.info(
                "Ad found at %.2fs - %.2fs (segments %d-%d)",
                seg_start,
                seg_end,
                block[0],
                block[-1],
            )
            logger.info("  Snapping to silence: %.2fs - %.2fs", final_start, final_end)
            final_ad_blocks.append(
                {
                    "start": final_start,
                    "end": final_end,
                    "text": ad_text,
                    "confidence": block_confidence,
                    "frequency": block_frequency,
                    "source": "llm",
                }
            )

        return final_ad_blocks

    def process_file(self, mp3_path: Path, suspicious: Set[int], force: bool = False, dry_run: bool = False, update: bool = False) -> str:
        """Phase 2: Detect ads and cut audio for a single file."""
        self._emit_progress("queued", mp3_path, phase="detection")
        sidecar = self._ensure_sidecar(mp3_path)
        is_success = sidecar.get("status") == "success"

        if is_success and not force and not update:
            logger.info("Skipping already processed file: %s", mp3_path.name)
            self._emit_progress("skipped", mp3_path, phase="detection", reason="already_success")
            return "skipped"

        if sidecar.get("status") == "new":
            logger.warning("Skipping file with status=new (transcription missing): %s", mp3_path.name)
            self._emit_progress("skipped", mp3_path, phase="detection", reason="missing_transcription")
            return "skipped"

        started_at = time.monotonic()
        # Use renamed "original_segments" key
        segments = self._segments_mem_cache.get(mp3_path) or sidecar.get("original_segments", [])
        if not segments:
            self._emit_progress("failed", mp3_path, phase="detection", reason="missing_segments")
            raise RuntimeError(f"No sidecar segments found for {mp3_path.name}")

        final_ad_blocks: List[dict]

        if update and is_success:
            logger.info("Updating from backup for: %s", mp3_path.name)
            # Restore from backup
            backup_path = sidecar.get("backup_path")
            backup_srt_path = sidecar.get("backup_srt_path")
            if not backup_path or not os.path.exists(backup_path):
                raise RuntimeError(f"Cannot update: backup_path missing or not found for {mp3_path.name}")
            
            if not dry_run:
                shutil.copy2(backup_path, mp3_path)
                if backup_srt_path and os.path.exists(backup_srt_path):
                    shutil.copy2(backup_srt_path, mp3_path.with_suffix(".srt"))
            
            final_ad_blocks = sidecar.get("ad_blocks", [])
        else:
            freq_indices = {idx for idx, seg in enumerate(segments) if _to_int(seg.get("frequency", 0)) > 1}
            all_suspicious = suspicious | freq_indices

            try:
                self._emit_progress("analyzing_llm", mp3_path, phase="detection")
                parsed, llm_payload = analyze_with_llm(self.config, segments, all_suspicious)
            except Exception as exc:
                logger.error("LLM failed for %s: %s", mp3_path.name, exc)
                self._emit_progress("failed", mp3_path, phase="detection", reason="llm_failed")
                return "skipped"
            
            parsed_summary = str(parsed.get("summary", ""))
            self._save_summary(mp3_path, parsed_summary)

            parsed_blocks = parse_llm_ad_blocks(
                parsed.get("ads", []),
                max_index=len(segments),
                min_confidence=self.config.min_confidence,
            )
            self._emit_progress("deriving_ad_blocks", mp3_path, phase="detection")
            final_ad_blocks = self._derive_ad_blocks(mp3_path, segments, parsed_blocks)

            sidecar["ad_blocks"] = final_ad_blocks
            self._save_sidecar(mp3_path)

        if not final_ad_blocks:
            logger.info("No ads detected (after confidence filter). Skipping cut.")
            if not dry_run:
                sidecar["status"] = "success"
                sidecar["processing_info"]["processed_at"] = utc_now_iso()
                sidecar["processing_info"]["processing_time_seconds"] = time.monotonic() - started_at
                sidecar["backup_path"] = None
                self._save_sidecar(mp3_path)
            self._emit_progress("processed", mp3_path, phase="detection", reason="no_ads")
            return "processed"

        cut_regions = [(_to_float(block.get("start")), _to_float(block.get("end"))) for block in final_ad_blocks]
        total_duration = get_audio_duration(mp3_path)
        total_ad_time = sum(end - start for start, end in cut_regions)
        ad_ratio = total_ad_time / total_duration if total_duration > 0 else 0
        if ad_ratio > 0.50:
            self._emit_progress("failed", mp3_path, phase="detection", reason="safety_guardrail")
            raise RuntimeError(
                f"SAFETY: Ad time ({total_ad_time:.1f}s) is {ad_ratio * 100:.0f}% of episode ({total_duration:.1f}s)"
            )

        if dry_run:
            logger.info(
                "DRY RUN: Would cut %.1fs of ads (%.0f%% of %.1fs) from %s",
                total_ad_time,
                ad_ratio * 100,
                total_duration,
                mp3_path.name,
            )
            for region_start, region_end in cut_regions:
                logger.info("  Would cut: %.2fs - %.2fs (%.1fs)", region_start, region_end, region_end - region_start)
            self._emit_progress("processed", mp3_path, phase="detection", reason="dry_run")
            return "processed"

        self._emit_progress("cutting_audio", mp3_path, phase="detection")
        try:
            backup_path = self._cut_audio(mp3_path, cut_regions, segments)
        except Exception:
            self._emit_progress("failed", mp3_path, phase="detection", reason="cut_failed")
            raise
        sidecar["status"] = "success"
        sidecar["backup_path"] = backup_path
        sidecar["backup_srt_path"] = str(Path(backup_path).with_suffix(".srt")) if backup_path else None
        sidecar["processing_info"]["processed_at"] = utc_now_iso()
        sidecar["processing_info"]["processing_time_seconds"] = time.monotonic() - started_at
        self._save_sidecar(mp3_path)
        self._emit_progress("processed", mp3_path, phase="detection")
        return "processed"

    def _cut_audio(self, mp3_path: Path, cut_regions: List[Tuple[float, float]], segments: List[dict]) -> str | None:
        out_path = mp3_path.with_name(f"{mp3_path.stem}-clean.mp3")
        total_duration = get_audio_duration(mp3_path)

        normalized_cut = normalize_cut_regions(cut_regions, total_duration)
        keep_regions = compute_keep_regions(normalized_cut, total_duration)
        if not keep_regions:
            raise RuntimeError("No keep regions left after cut planning")

        filter_parts = []
        concat_inputs = []
        for i, (start, end) in enumerate(keep_regions):
            filter_parts.append(f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")

        filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(keep_regions)}:v=0:a=1[outa]")
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-i",
            str(mp3_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(out_path),
        ]
        logger.info("Executing robust cut with re-encode...")
        subprocess.run(cmd, check=True)

        backup_path: str | None = None
        if self.config.backup_enabled:
            # Backup original mp3/SRT to configured backup location.
            backup_dir = Path(self.config.backup_location).expanduser()
            if not backup_dir.is_absolute():
                backup_dir = (mp3_path.parent / backup_dir).resolve()
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir / mp3_path.name
            backup_srt = backup_dir / mp3_path.with_suffix(".srt").name

            shutil.copy2(mp3_path, backup_file)
            srt_path = mp3_path.with_suffix(".srt")
            if srt_path.exists():
                shutil.copy2(srt_path, backup_srt)
            backup_path = str(backup_file.absolute())

        shutil.move(out_path, mp3_path)

        shifted_segments = shift_transcript(segments, keep_regions)
        save_srt(shifted_segments, mp3_path.with_suffix(".srt"))
        logger.info("Done. Clean file and updated transcript are at %s", mp3_path)

        return backup_path


def should_process_mp3(path: Path, excluded_dirs: Set[Path] | None = None) -> bool:
    if path.suffix.lower() != ".mp3":
        return False
    if "backup" in path.parts:
        return False
    if excluded_dirs:
        try:
            resolved_path = path.resolve()
        except OSError:
            resolved_path = path.absolute()
        for excluded_dir in excluded_dirs:
            try:
                resolved_path.relative_to(excluded_dir)
                return False
            except ValueError:
                continue
    name = path.name.lower()
    return not (name.endswith("-backup.mp3") or name.endswith("-clean.mp3"))


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    progress_mode = resolve_progress_mode(
        progress_flag=args.progress,
        stderr_is_tty=sys.stderr.isatty(),
        tqdm_available=tqdm is not None,
    )
    log_file_override = Path(args.log_file).expanduser() if args.log_file else None
    configure_logging(use_tqdm_console=(progress_mode == "tqdm"), log_file=log_file_override)
    progress = RunProgressManager(mode=progress_mode)
    run_started_at = time.monotonic()

    if not args.path:
        parser.print_help()
        progress.close()
        return

    path = Path(args.path)
    if not path.exists():
        logger.error("Path not found: %s", path)
        progress.close()
        sys.exit(1)

    def on_processor_progress(event: str, mp3_path: Path | None, _payload: Dict[str, Any]):
        progress.set_status(mp3_path, event)

    processor = Processor(args.config, progress_callback=on_processor_progress)

    targets: List[Path] = []
    excluded_dirs: Set[Path] = set()
    configured_backup_dir = Path(processor.config.backup_location).expanduser()
    if not configured_backup_dir.is_absolute():
        base_dir = path.parent if path.is_file() else path
        configured_backup_dir = (base_dir / configured_backup_dir).resolve()
    else:
        configured_backup_dir = configured_backup_dir.resolve()
    excluded_dirs.add(configured_backup_dir)

    if path.is_file() and should_process_mp3(path, excluded_dirs=excluded_dirs):
        targets.append(path)
    elif path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and should_process_mp3(candidate, excluded_dirs=excluded_dirs):
                targets.append(candidate)

    targets.sort()
    logger.info("Found %d files to process.", len(targets))

    for target in targets:
        processor._ensure_sidecar(target)

    def needs_processing(mp3_path: Path) -> bool:
        sidecar = processor._ensure_sidecar(mp3_path)
        return args.force or args.update or sidecar.get("status") != "success"

    # Phase 1: Transcribe all files
    logger.info("=== Phase 1: Transcription ===")
    progress.start_phase("Phase 1: Transcription", total=len(targets))
    phase1_failed = 0
    for target in targets:
        item_started = time.monotonic()
        progress.start_item(target)
        try:
            processor.ensure_transcription(target)
        except Exception as exc:
            phase1_failed += 1
            progress.set_status(target, "failed")
            logger.error("Transcription failed for %s: %s", target.name, exc)
        finally:
            progress.complete_item(time.monotonic() - item_started)

    if args.transcribe_only:
        progress.close()
        logger.info(
            "--transcribe-only flag detected. Transcription complete. failed=%d elapsed=%s",
            phase1_failed,
            _format_duration(time.monotonic() - run_started_at),
        )
        return

    logger.info("=== Segment Frequency ===")
    processor.update_segment_frequencies(targets)

    # Preload segments to memory for massive I/O speedup
    logger.info("=== Preloading Data ===")
    processor.preload_all_segments(targets)

    new_targets = {t for t in targets if needs_processing(t)}
    logger.info("%d files identified as new/unprocessed.", len(new_targets))

    # Cross-file matching
    logger.info("=== Cross-File Matching ===")
    cross_started = False

    def on_fuzzy_progress(payload: Dict[str, Any]):
        nonlocal cross_started
        chunks_total = int(payload.get("chunks_total", 0))
        if not cross_started:
            progress.start_cross_file(chunks_total)
            cross_started = True
        progress.update_cross_file(
            chunks_done=int(payload.get("chunks_done", 0)),
            chunks_total=chunks_total,
            comparisons=int(payload.get("comparisons", 0)),
            matches=int(payload.get("matches", 0)),
            elapsed=float(payload.get("elapsed", 0.0)),
        )

    suspicious = find_repeated_segments(
        targets,
        new_targets,
        processor.config.sim_thresh, 
        load_all_segments_func=processor.load_all_segments,
        fuzzy_progress_callback=on_fuzzy_progress,
    )

    # Phase 2: Detect & cut
    logger.info("=== Phase 2: Detection & Cutting ===")
    processed = 0
    skipped = 0
    failed = phase1_failed
    phase2_targets = [t for t in targets if needs_processing(t)]
    progress.start_phase("Phase 2: Detection & Cutting", total=len(phase2_targets))

    for target in targets:
        if not needs_processing(target):
            skipped += 1
            logger.info("Skipping already successful file: %s", target.name)
            continue

        item_started = time.monotonic()
        progress.start_item(target)
        try:
            status = processor.process_file(
                target,
                suspicious=suspicious.get(target, set()),
                force=args.force,
                dry_run=args.dry_run,
                update=args.update,
            )
        except Exception as exc:
            failed += 1
            progress.set_status(target, "failed")
            logger.error("Processing failed for %s: %s", target.name, exc)
            progress.complete_item(time.monotonic() - item_started)
            continue

        progress.complete_item(time.monotonic() - item_started)
        if status == "processed":
            processed += 1
        else:
            skipped += 1

    progress.close()
    logger.info(
        "Run complete. processed=%d skipped=%d failed=%d elapsed=%s",
        processed,
        skipped,
        failed,
        _format_duration(time.monotonic() - run_started_at),
    )


if __name__ == "__main__":
    main()
