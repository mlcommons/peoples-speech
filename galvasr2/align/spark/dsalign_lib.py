import collections
import io
import json
import signal
import threading
from typing import Dict, List, Tuple

from collections import Counter
from search import FuzzySearch
import textdistance
from glob import glob
from text import Alphabet, TextCleaner, levenshtein, similarity
from utils import enweight
from dsalign_main import ALGORITHMS, NAMED_NUMBERS

import logging

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T

from galvasr2.align.spark.timeout import timeout

def parse_ctm(ctm_buffer, silence_words: set):
  fh = io.StringIO(ctm_buffer)
  # utterance, channel, start time in sec, duration in sec, label
  # 1 A 2.560 0.016 COMMISSION
  Fragment = collections.namedtuple("Fragment", ["start_ms", "end_ms", "word"])
  words = []
  for line in fh:
    try:
      _, _, start_s, duration_s, word, _ = line.split(" ")
    except ValueError:
      print("GALVEZ: problematic line: ", line)
      return []
    if word in silence_words:
      continue

    start_ms = int(float(start_s) * 1000)
    duration_ms = int(float(duration_s) * 1000)
    words.append(Fragment(start_ms=start_ms,
                          end_ms=start_ms + duration_ms,
                          word=word))
  return words

def join_fragments(ctm_contents: List, max_length_ms: int,
                   max_silence_length_ms: int) -> Dict:
    fragments = []
    if len(ctm_contents) > 0:
        start_ms = ctm_contents[0].start_ms
        transcript = [ctm_contents[0].word]
    for i in range(1, len(ctm_contents)):
        current_word_fragment = ctm_contents[i]
        previous_word_fragment = ctm_contents[i - 1]
        silence_duration_ms = current_word_fragment.start_ms - previous_word_fragment.end_ms
        if (current_word_fragment.end_ms - start_ms > max_length_ms or
            silence_duration_ms > max_silence_length_ms):
            fragments.append({
                "start": start_ms,
                "end": previous_word_fragment.end_ms,
                "transcript": " ".join(transcript)})
            start_ms = current_word_fragment.start_ms
            transcript = []
        transcript.append(current_word_fragment.word)
    # Add final straggler fragment, if it exists
    if start_ms != -1:
        fragments.append({
            "start": start_ms,
            "end": ctm_contents[-1].end_ms,
            "transcript": " ".join(transcript)})
    for fragment in fragments:
        assert fragment["end"] - fragment["start"] <= max_length_ms
    return fragments


def align(args, asr_output_fragments, transcript, alphabet):
  # asr_output, transcript, aligned = triple

  tc = TextCleaner(alphabet,
                   dashes_to_ws=not args.text_keep_dashes,
                   normalize_space=not args.text_keep_ws,
                   to_lower=not args.text_keep_casing)
  tc.add_original_text(transcript)

  # Chossing these values may speed up search?
  search = FuzzySearch(tc.clean_text,
                       max_candidates=args.align_max_candidates,
                       candidate_threshold=args.align_candidate_threshold,
                       match_score=args.align_match_score,
                       mismatch_score=args.align_mismatch_score,
                       gap_score=args.align_gap_score)

  # logging.debug("Loading transcription log from %s..." % tlog)
  # with open(tlog, 'r', encoding='utf-8') as transcription_log_file:
  #   fragments = json.load(transcription_log_file)
  fragments = asr_output_fragments
  end_fragments = (args.start + args.num_samples) if args.num_samples else len(fragments)
  fragments = fragments[args.start:end_fragments]
  for index, fragment in enumerate(fragments):
    meta = {}
    for key, value in list(fragment.items()):
      if key not in ['start', 'end', 'transcript']:
        meta[key] = value
        del fragment[key]
    fragment['meta'] = meta
    fragment['index'] = index
    # is strip() necessary?
    fragment['transcript'] = fragment['transcript'].strip()

  reasons = Counter()

  def skip(index, reason):
    logging.info('Fragment {}: {}'.format(index, reason))
    reasons[reason] += 1

  def split_match(fragments, start=0, end=-1):
    n = len(fragments)
    # print(f"GALVEZ: n={n}")
    # import sys; sys.stdout.flush()
    if n < 1:
      return
    elif n == 1:
      weighted_fragments = [(0, fragments[0])]
    else:
      # so we later know the original index of each fragment
      weighted_fragments = enumerate(fragments)
      # assigns high values to long statements near the center of the list
      weighted_fragments = enweight(weighted_fragments)
      weighted_fragments = map(lambda fw: (fw[0], (1 - fw[1]) * len(fw[0][1]['transcript'])), weighted_fragments)
      # fragments with highest weights first
      weighted_fragments = sorted(weighted_fragments, key=lambda fw: fw[1], reverse=True)
      # strip weights
      weighted_fragments = list(map(lambda fw: fw[0], weighted_fragments))
    for index, fragment in weighted_fragments:
      # find_best?
      match = search.find_best(fragment['transcript'], start=start, end=end)
      match_start, match_end, sws_score, match_substitutions = match
      # At least half must overlap...
      # print(f"GALVEZ: sws_score={sws_score}")
      # import sys; sys.stdout.flush()
      if sws_score > (n - 1) / (2 * n):
        #print(f"GALVEZ: sws passed sws_score={sws_score}")
        # import sys; sys.stdout.flush()
        fragment['match-start'] = match_start
        fragment['match-end'] = match_end
        fragment['sws'] = sws_score
        fragment['substitutions'] = match_substitutions
        # Here's the recursive joining, is that right?
        # What does this do?
        for f in split_match(fragments[0:index], start=start, end=match_start):
          yield f
        yield fragment
        for f in split_match(fragments[index + 1:], start=match_end, end=end):
          yield f
        return
    for _, _ in weighted_fragments:
      yield None

  # What is a matched fragment?
  # Are matched fragments ever joined?
  matched_fragments = split_match(fragments)
  matched_fragments = list(filter(lambda f: f is not None, matched_fragments))

  # print(f"GALVEZ:matched_fragments_length={len(matched_fragments)}")

  similarity_algos = {}

  def phrase_similarity(algo, a, b):
    if algo in similarity_algos:
      return similarity_algos[algo](a, b)
    algo_impl = lambda aa, bb: None
    if algo.lower() == 'wng':
      algo_impl = similarity_algos[algo] = lambda aa, bb: similarity(
        aa,
        bb,
        direction=1,
        min_ngram_size=args.align_wng_min_size,
        max_ngram_size=args.align_wng_max_size,
        size_factor=args.align_wng_size_factor,
        position_factor=args.align_wng_position_factor)
    elif algo in ALGORITHMS:
      algo_impl = similarity_algos[algo] = getattr(textdistance, algo).normalized_similarity
    else:
      logging.fatal('Unknown similarity metric "{}"'.format(algo))
      exit(1)
    return algo_impl(a, b)

  def get_similarities(a, b, n, gap_text, gap_meta, direction):
    if direction < 0:
      a, b, gap_text, gap_meta = a[::-1], b[::-1], gap_text[::-1], gap_meta[::-1]
    similarities = list(map(
      lambda i: (args.align_word_snap_factor if gap_text[i + 1] == ' ' else 1) *
      (args.align_phrase_snap_factor if gap_meta[i + 1] is None else 1) *
      (phrase_similarity(args.align_similarity_algo, a, b + gap_text[1:i + 1])),
      range(n)))
    best = max((v, i) for i, v in enumerate(similarities))[1] if n > 0 else 0
    return best, similarities

  for index in range(len(matched_fragments) + 1):
    if index > 0:
      a = matched_fragments[index - 1]
      a_start, a_end = a['match-start'], a['match-end']
      a_len = a_end - a_start
      a_stretch = int(a_len * args.align_stretch_fraction)
      a_shrink = int(a_len * args.align_shrink_fraction)
      a_end = a_end - a_shrink
      a_ext = a_shrink + a_stretch
    else:
      a = None
      a_start = a_end = 0
    if index < len(matched_fragments):
      b = matched_fragments[index]
      b_start, b_end = b['match-start'], b['match-end']
      b_len = b_end - b_start
      b_stretch = int(b_len * args.align_stretch_fraction)
      b_shrink = int(b_len * args.align_shrink_fraction)
      b_start = b_start + b_shrink
      b_ext = b_shrink + b_stretch
    else:
      b = None
      b_start = b_end = len(search.text)

    assert a_end <= b_start
    assert a_start <= a_end
    assert b_start <= b_end
    if a_end == b_start or a_start == a_end or b_start == b_end:
      continue
    gap_text = tc.clean_text[a_end - 1:b_start + 1]
    gap_meta = tc.meta[a_end - 1:b_start + 1]

    if a:
      a_best_index, a_similarities = get_similarities(a['transcript'],
                                                      tc.clean_text[a_start:a_end],
                                                      min(len(gap_text) - 1, a_ext),
                                                      gap_text,
                                                      gap_meta,
                                                      1)
      a_best_end = a_best_index + a_end
    if b:
      b_best_index, b_similarities = get_similarities(b['transcript'],
                                                      tc.clean_text[b_start:b_end],
                                                      min(len(gap_text) - 1, b_ext),
                                                      gap_text,
                                                      gap_meta,
                                                      -1)
      b_best_start = b_start - b_best_index

    if a and b and a_best_end > b_best_start:
      overlap_start = b_start - len(b_similarities)
      a_similarities = a_similarities[overlap_start - a_end:]
      b_similarities = b_similarities[:len(a_similarities)]
      best_index = max((sum(v), i) for i, v in enumerate(zip(a_similarities, b_similarities)))[1]
      a_best_end = b_best_start = overlap_start + best_index

      if a:
        a['match-end'] = a_best_end
      if b:
        b['match-start'] = b_best_start

  def apply_number(number_key, index, fragment, show, get_value):
    kl = number_key.lower()
    should_output = getattr(args, 'output_' + kl)
    min_val, max_val = getattr(args, 'output_min_' + kl), getattr(args, 'output_max_' + kl)
    if kl.endswith('len') and min_val is None:
      min_val = 1
    if should_output or min_val or max_val:
      val = get_value()
      if not kl.endswith('len'):
        show.insert(0, '{}: {:.2f}'.format(number_key, val))
        if should_output:
          # So it's part of the metadata? Hmm... If I could query you 
          fragment[kl] = val
      reason_base = '{} ({})'.format(NAMED_NUMBERS[number_key][0], number_key)
      reason = None
      if min_val and val < min_val:
        reason = reason_base + ' too low'
      elif max_val and val > max_val:
        reason = reason_base + ' too high'
      if reason:
        skip(index, reason)
        return True
    return False

  substitutions = Counter()
  result_fragments = []
  # print(f"GALVEZ:matched_fragments_length2={len(matched_fragments)}")
  for fragment in matched_fragments:
    index = fragment['index']
    time_start = fragment['start']
    time_end = fragment['end']
    fragment_transcript = fragment['transcript']
    result_fragment = {
      'start': time_start,
      'end': time_end
    }
    sample_numbers = []

    if apply_number('tlen', index, result_fragment, sample_numbers, lambda: len(fragment_transcript)):
      continue
    result_fragment['transcript'] = fragment_transcript

    if 'match-start' not in fragment or 'match-end' not in fragment:
      skip(index, 'No match for transcript')
      continue
    match_start, match_end = fragment['match-start'], fragment['match-end']
    if match_end - match_start <= 0:
      skip(index, 'Empty match for transcript')
      continue
    original_start = tc.get_original_offset(match_start)
    original_end = tc.get_original_offset(match_end)
    result_fragment['text-start'] = original_start
    result_fragment['text-end'] = original_end

    meta_dict = {}
    for meta in list(tc.collect_meta(match_start, match_end)) + [fragment['meta']]:
      for key, value in meta.items():
        if key == 'text':
          continue
        if key in meta_dict:
          values = meta_dict[key]
        else:
          values = meta_dict[key] = []
        if value not in values:
          values.append(value)
    result_fragment['meta'] = meta_dict

    result_fragment['aligned-raw'] = tc.original_text[original_start:original_end]

    fragment_matched = tc.clean_text[match_start:match_end]
    if apply_number('mlen', index, result_fragment, sample_numbers, lambda: len(fragment_matched)):
      continue
    result_fragment['aligned'] = fragment_matched

    if apply_number('SWS', index, result_fragment, sample_numbers, lambda: 100 * fragment['sws']):
      continue

    should_skip = False
    for algo in ALGORITHMS:
      should_skip = should_skip or apply_number(algo, index, result_fragment, sample_numbers,
                                                lambda: 100 * phrase_similarity(algo,
                                                                                fragment_matched,
                                                                                fragment_transcript))
    if should_skip:
      continue

    if apply_number('CER', index, result_fragment, sample_numbers,
                    lambda: 100 * levenshtein(fragment_transcript, fragment_matched) /
                    len(fragment_matched)):
      continue

    if apply_number('WER', index, result_fragment, sample_numbers,
                    lambda: 100 * levenshtein(fragment_transcript.split(), fragment_matched.split()) /
                    len(fragment_matched.split())):
      continue

    substitutions += fragment['substitutions']

    result_fragments.append(result_fragment)
    # Why don't I see this output?
    logging.debug('Fragment %d aligned with %s' % (index, ' '.join(sample_numbers)))
    # T is for transcript
    logging.debug('- T: ' + args.text_context * ' ' + '"%s"' % fragment_transcript)
    # O is for original
    logging.debug('- O: %s|%s|%s' % (
      tc.clean_text[match_start - args.text_context:match_start],
      fragment_matched,
      tc.clean_text[match_end:match_end + args.text_context]))
    # if args.play:
    #   # this seems like a really bad alignment
    #   # DEBUG:root:Fragment 869 aligned with
    #   # DEBUG:root:- T:           "angus heard"
    #   # DEBUG:root:- O: as to give| us heart| to believ
            
    #   # trim feature of sox seems useful.
    #   subprocess.check_call(['play',
    #                          '--no-show-progress',
    #                          args.audio,
    #                          'trim',
    #                          str(time_start / 1000.0),
    #                          '=' + str(time_end / 1000.0)])
      # Playing audio seems interesting.
      # Should pause after each playtime.
  # with open(aligned, 'w', encoding='utf-8') as result_file:
  #   result_file.write(json.dumps(result_fragments, indent=4 if args.output_pretty else None, ensure_ascii=False))
  num_result_fragments = len(result_fragments)
  num_dropped_fragments = len(fragments) - len(result_fragments)
  # print(f"GALVEZ:reasons={reasons}")
  # print(f"GALVEZ:num_result_fragments={num_result_fragments}")
  # print(f"GALVEZ:num_dropped_fragments={num_dropped_fragments}")
  return num_result_fragments, num_dropped_fragments, reasons, result_fragments

def prepare_align_udf(dsalign_args, alphabet_path):
  args = dsalign_args
  ALIGN_RETURN_TYPE = T.StructType([T.StructField("start_ms", T.ArrayType(T.LongType())),
                                    T.StructField("end_ms", T.ArrayType(T.LongType())),
                                    T.StructField("label", T.ArrayType(T.StringType()))])
  @F.pandas_udf(ALIGN_RETURN_TYPE)
  def align_table(name_series: pd.Series,
                  audio_name_series: pd.Series,
                  transcript_series: pd.Series,
                  ctm_content_series: pd.Series) -> pd.DataFrame:
    alphabet = Alphabet(alphabet_path)
    silence_words = frozenset(["<unk>", "[laughter]", "[noise]"])
    result_dict = {"start_ms": [],
                   "end_ms": [],
                   "label": []}
    for name, audio_name, ctm_content, transcript in zip(name_series, audio_name_series,
                                                         ctm_content_series, transcript_series):
      # print(f"GALVEZ:name={name}")
      # print(f"GALVEZ:audio_name={audio_name}")
      fragments = join_fragments(parse_ctm(ctm_content, silence_words), 15_000, 3_000)
      # timeout after 200 seconds
      output = timeout(align, (args, fragments, transcript, alphabet),
                       timeout_duration=800)
      if output is None:
        print(f"GALVEZ: timed out for name={name} audio_name={audio_name}")
      if output is not None:
        _, _, _, aligned_results = output
        start_times = []
        end_times = []
        labels = []
        for result in aligned_results:
          start_times.append(result['start'])
          end_times.append(result['end'])
          labels.append(result['aligned-raw'])
          # TODO: Add metrics like CER, WER, etc., for filtering out
          # bad alignments later.
        result_dict["start_ms"].append(start_times)
        result_dict["end_ms"].append(end_times)
        result_dict["label"].append(labels)
      else:
        result_dict["start_ms"].append([])
        result_dict["end_ms"].append([])
        result_dict["label"].append([])
    return pd.DataFrame(result_dict)
  return align_table
