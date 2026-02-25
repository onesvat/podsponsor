import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podsponsor import (
    ManifestStore,
    Processor,
    compute_keep_regions,
    find_repeated_segments,
    load_words_json,
    normalize_cut_regions,
    parse_llm_ad_blocks,
    save_words_json,
    expand_ad_indices,
    shift_transcript,
)


class ManifestTests(unittest.TestCase):
    def test_success_entry_skips_until_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            show_dir = Path(tmp)
            mp3_path = show_dir / "episode.mp3"
            mp3_path.write_bytes(b"abc")

            manifest = ManifestStore(show_dir, ".podsponsor-manifest.json")
            fingerprint = manifest.fingerprint(mp3_path)
            manifest.mark_success(mp3_path, fingerprint)

            self.assertTrue(manifest.should_skip(mp3_path, force=False))
            self.assertFalse(manifest.should_skip(mp3_path, force=True))

            mp3_path.write_bytes(b"abcdef")
            self.assertFalse(manifest.should_skip(mp3_path, force=False))


class CutRegionTests(unittest.TestCase):
    def test_normalize_and_keep_regions(self):
        cuts = [(-1, 2), (10, 20), (15, 25), (40, 41)]
        normalized = normalize_cut_regions(cuts, total_duration=50)
        self.assertEqual([(0.0, 2.0), (10.0, 25.0), (40.0, 41.0)], normalized)

        keep = compute_keep_regions(normalized, total_duration=50)
        self.assertEqual([(2.0, 10.0), (25.0, 40.0), (41.0, 50.0)], keep)


class WordsJsonTests(unittest.TestCase):
    def test_save_and_load_words_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            words_path = Path(tmp) / "episode.words.json"
            segments = [
                {
                    "start": 1.0,
                    "end": 3.5,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 1.0, "end": 1.5},
                        {"word": "world", "start": 1.6, "end": 3.5},
                    ],
                },
                {
                    "start": 4.0,
                    "end": 6.0,
                    "text": "Second segment",
                },
            ]
            save_words_json(segments, words_path)
            loaded = load_words_json(words_path)

            self.assertEqual(2, len(loaded))
            self.assertEqual("Hello world", loaded[0]["text"])
            self.assertEqual(2, len(loaded[0]["words"]))
            self.assertAlmostEqual(1.0, loaded[0]["words"][0]["start"])
            self.assertNotIn("words", loaded[1])  # No words key if not present

    def test_load_missing_file(self):
        segments = load_words_json(Path("/nonexistent/path.words.json"))
        self.assertEqual([], segments)


class CrossFileMatchingTests(unittest.TestCase):
    def test_finds_repeated_segments(self):
        segments_a = [
            {"start": 0, "end": 5, "text": "Welcome to our show today"},
            {"start": 5, "end": 10, "text": "This episode is brought to you by Acme Corp visit acme.com"},
            {"start": 10, "end": 15, "text": "Now let's talk about science"},
        ]
        segments_b = [
            {"start": 0, "end": 5, "text": "Hello everyone and welcome"},
            {"start": 5, "end": 10, "text": "This episode is brought to you by Acme Corp visit acme.com"},
            {"start": 10, "end": 15, "text": "Today we discuss history"},
        ]
        all_segs = {Path("/a.mp3"): segments_a, Path("/b.mp3"): segments_b}
        suspicious = find_repeated_segments(all_segs, sim_threshold=0.80)

        self.assertIn(1, suspicious.get(Path("/a.mp3"), set()))
        self.assertIn(1, suspicious.get(Path("/b.mp3"), set()))
        self.assertNotIn(0, suspicious.get(Path("/a.mp3"), set()))
        self.assertNotIn(2, suspicious.get(Path("/a.mp3"), set()))

    def test_no_repeats_across_single_file(self):
        segments = [
            {"start": 0, "end": 5, "text": "This is a unique segment number one"},
            {"start": 5, "end": 10, "text": "This is a unique segment number two"},
        ]
        all_segs = {Path("/a.mp3"): segments}
        suspicious = find_repeated_segments(all_segs, sim_threshold=0.80)
        self.assertEqual(0, sum(len(v) for v in suspicious.values()))


class ParseLlmAdBlocksTests(unittest.TestCase):
    def test_parses_valid_blocks(self):
        raw = [
            {"start_index": 5, "end_index": 8, "confidence": 0.95},
            {"start_index": 20, "end_index": 22, "confidence": 0.85},
        ]
        blocks = parse_llm_ad_blocks(raw, max_index=30, min_confidence=0.70)
        self.assertEqual(2, len(blocks))
        self.assertEqual((5, 8, 0.95), blocks[0])
        self.assertEqual((20, 22, 0.85), blocks[1])

    def test_filters_low_confidence(self):
        raw = [
            {"start_index": 5, "end_index": 8, "confidence": 0.95},
            {"start_index": 20, "end_index": 22, "confidence": 0.50},
        ]
        blocks = parse_llm_ad_blocks(raw, max_index=30, min_confidence=0.70)
        self.assertEqual(1, len(blocks))
        self.assertEqual((5, 8, 0.95), blocks[0])

    def test_filters_out_of_range(self):
        raw = [
            {"start_index": 5, "end_index": 50, "confidence": 0.95},  # end > max
            {"start_index": -1, "end_index": 3, "confidence": 0.95},  # negative start
        ]
        blocks = parse_llm_ad_blocks(raw, max_index=30, min_confidence=0.70)
        self.assertEqual(0, len(blocks))

    def test_handles_non_list(self):
        blocks = parse_llm_ad_blocks("not a list", max_index=30, min_confidence=0.70)
        self.assertEqual(0, len(blocks))


class NeighbourExpansionTests(unittest.TestCase):
    def test_expands_only_suspicious_neighbours(self):
        ad_blocks = [(5, 7, 0.95)]
        suspicious = {3, 4, 5, 6, 7, 8}  # 3,4 before; 8 after are suspicious
        result = expand_ad_indices(ad_blocks, suspicious, total_segments=20, expand_range=2)

        # Original block
        self.assertIn(5, result)
        self.assertIn(6, result)
        self.assertIn(7, result)
        # Expanded (suspicious neighbours)
        self.assertIn(4, result)
        self.assertIn(3, result)
        self.assertIn(8, result)
        # Not expanded (not suspicious)
        self.assertNotIn(9, result)

    def test_no_expansion_without_suspicious(self):
        ad_blocks = [(5, 7, 0.95)]
        suspicious = set()  # No suspicious neighbours
        result = expand_ad_indices(ad_blocks, suspicious, total_segments=20, expand_range=2)

        self.assertEqual({5, 6, 7}, result)


class ProcessorFlowTests(unittest.TestCase):
    def test_existing_srt_skips_transcribe_when_words_json_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                """
whisper:
  model: medium
  device: cpu
  compute_type: float32
llm:
  summary_language: en
  prompt: "You are an expert podcast analyst..."
  providers:
    - base_url: http://localhost:11434/v1
      model: test
      api_key: dummy
      timeout_seconds: 5
      max_retries: 0
detection:
  similarity_threshold: 0.8
  silence_db: -40
  silence_min_duration: 1.5
  min_confidence: 0.70
  min_ad_duration: 8.0
processing:
  output_scheme: save_as_clean
  manifest_name: .podsponsor-manifest.json
  local_ads_name: ads.json
""".strip(),
                encoding="utf-8",
            )

            episode_mp3 = tmp_path / "episode.mp3"
            episode_mp3.write_bytes(b"fake-mp3")

            # Create both SRT and words.json
            episode_srt = tmp_path / "episode.srt"
            episode_srt.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nhello world\n",
                encoding="utf-8",
            )
            words_path = tmp_path / "episode.words.json"
            save_words_json(
                [{"start": 1.0, "end": 2.0, "text": "hello world"}],
                words_path,
            )

            processor = Processor(str(config_path))
            with patch.object(processor.transcriber, "transcribe", side_effect=AssertionError("transcribe should not run")):
                processor.ensure_transcription(episode_mp3)

            # Now test process_file with mocked LLM
            with patch("podsponsor.analyze_with_llm", return_value={"ads": [], "summary": "ok"}):
                status = processor.process_file(episode_mp3, suspicious=set(), force=True)

            self.assertEqual("processed", status)
            self.assertTrue((tmp_path / "episode.md").exists())


class ShiftTranscriptTests(unittest.TestCase):
    def test_shift_transcript_removes_cut_words(self):
        segments = [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "Before the ad ",
                "words": [
                    {"word": "Before", "start": 0.0, "end": 1.0},
                    {"word": "the", "start": 1.1, "end": 2.0},
                    {"word": "ad", "start": 2.1, "end": 3.0},
                    {"word": "starts", "start": 8.1, "end": 9.0}, # this one is in cut region
                ]
            }
        ]
        
        keep_regions = [(0.0, 5.0), (10.0, 20.0)]
        shifted = shift_transcript(segments, keep_regions)
        
        self.assertEqual(1, len(shifted))
        # "starts" should be removed because it was between 5.0 and 10.0
        self.assertEqual(3, len(shifted[0]["words"]))
        self.assertEqual("Before", shifted[0]["words"][0]["word"])
        self.assertEqual("ad", shifted[0]["words"][2]["word"])
        # Text should be rebuilt
        self.assertEqual("Beforethead", shifted[0]["text"])

    def test_shift_transcript_adjusts_times(self):
        segments = [
            {
                "start": 15.0,
                "end": 17.0,
                "text": "After the ad",
                "words": [
                    {"word": "After", "start": 15.0, "end": 16.0},
                ]
            }
        ]
        
        # Audio was cut from 5.0 to 10.0. So offset is 5.0 for everything after 10.0.
        keep_regions = [(0.0, 5.0), (10.0, 20.0)]
        shifted = shift_transcript(segments, keep_regions)
        
        self.assertEqual(1, len(shifted))
        self.assertAlmostEqual(10.0, shifted[0]["words"][0]["start"])
        self.assertAlmostEqual(11.0, shifted[0]["words"][0]["end"])
        self.assertAlmostEqual(10.0, shifted[0]["start"])
        self.assertAlmostEqual(11.0, shifted[0]["end"])


class FileFilterTests(unittest.TestCase):
    def test_should_process_mp3(self):
        from podsponsor import should_process_mp3
        self.assertTrue(should_process_mp3(Path("/some/dir/episode.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode.txt")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode-backup.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode-clean.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/backup/episode.mp3")))
        self.assertFalse(should_process_mp3(Path("/backup/folder/ep.mp3")))


if __name__ == "__main__":
    unittest.main()
