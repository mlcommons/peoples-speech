# Design
========

This forced aligner is (initially a fork of DSAlign). You cna read
more about it in that repo's algo.md file.

Roughly, the forced alignment process looks like this:

1. Do voice activity detection to chunk up each piece of audio into
smaller segments. This is not generally necessary, but because we
don't support streaming speech recognition at the moment with our
lingvo models, we must load each chunk of audio entirely into
memory. This code is in galvasr/align/audio.py. I have found that the
voice activity detector there is fairly over-reactive, such that audio
chunks are too small, unless an aggressiveness score of 0 (the lowest)
is chosen. (When the lowest is chosen, the average segment is 15
seconds in length, based on eyeball statistics). Vineel has suggested
a greedy algorithm where we join adjacent segments until each
"supersegment" is at least 15 seconds long. I agree that this is
probably a better approach.

1. Do automatic speech recognition. This is the most compute-intensive
part, and our current bottleneck. Note that you can optionally use a
language model "overtrained" on the text of the document you are
trying to align during this stage, which I recommend.

1. Do the recursive alignment algorithm. I recommend reading DSAlign's
doc/algo.md for this. It is a fairly good approach, and I have few
qualms with it currently.

The current problem with DSAlign is that it fully processes one full
audio-text pair (this is a document fulfilling the concept of
"audio-text pair":
https://archive.org/details/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49)
at a time, before reaching the next one. This eliminates the
possibility of batching, which would increase our compute throughput.

Based on a profiling by Max (https://pastebin.com/96kq59Mp), it is
clear that running the speech recognition model takes >90% of the time
(60.335/65.682).

```
   Ordered by: internal time
 
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       51   60.335    1.183   60.335    1.183 {built-in method deepspeech._impl.SpeechToText}
      435    1.699    0.004    2.178    0.005 edit_based.py:87(_cicled)
   716874    0.344    0.000    0.404    0.000 base.py:100(_ident)
   115792    0.308    0.000    0.666    0.000 phonetic.py:118(r_cost)
   <manually-cut-lines>
        1    0.002    0.002   64.998   64.998 align.py:368(main)
```

At the time of writing, we have a CPU quota of 24. I noticed it took
about 24 minutes to align a 30 minute audio file. Using DSAlign as it
is to align 10,000 hours of data will therefore take 

```
(((24 min / 30 min) * 100_000 hours) / 24 hours in a day) / 24 cpus = 139 days
```

This is unacceptably slow.

## Deeper Dive
==============

### Speeding up the ASR system

One obvious way to speed this up is to use batching, which increases
throughput. However, I discovered that DSAlign's (speech recognition
inference
library)[https://github.com/mozilla/DeepSpeech/blob/2e9c281d06ea8da97f7e4eebd3e4476350e7776a/native_client/deepspeech.h#L189]
does not support a batch size greater than 1. Therefore this
possibility is not available to us without changing to a different ASR
implementation.

It is also worthwhile to note that python's profiler does not
distinguish between parts of the native code. Therefore, I tried to
check using the "perf" linux profiler whether the acoustic model or
the language model was the bottleneck:


```
Samples: 6M of event 'cpu-clock', Event count (approx.): 1547209750000                                                                                                                                      
Overhead  Command  Shared Object                                      Symbol                                                                                                                                
  23.05%  python   libdeepspeech.so                                   [.] 0x00000000006b6d72
  11.43%  python   libdeepspeech.so                                   [.] 0x00000000006b6d8e
   8.99%  python   libdeepspeech.so                                   [.] 0x00000000006b6dcb
   6.05%  python   libdeepspeech.so                                   [.] 0x00000000006b6dab
   3.00%  python   python3.7                                          [.] _PyEval_EvalFrameDefault
   1.57%  python   libdeepspeech.so                                   [.] 0x00000000006b6d92
   1.57%  python   libdeepspeech.so                                   [.] 0x00000000006b6d76
   1.35%  python   libdeepspeech.so                                   [.] 0x00000000006b6dcf
   1.29%  python   libdeepspeech.so                                   [.] 0x00000000006b6d96
   1.24%  python   libdeepspeech.so                                   [.] 0x00000000006b6da3
   1.17%  python   libdeepspeech.so                                   [.] 0x00000000006b6d7b
   1.17%  python   libdeepspeech.so                                   [.] 0x00000000006b6daf
   1.13%  python   libdeepspeech.so                                   [.] 0x00000000006b6dd3
   1.13%  python   libdeepspeech.so                                   [.] 0x00000000006b6de9
   1.12%  python   libdeepspeech.so                                   [.] 0x00000000006b6d89
   1.08%  python   libdeepspeech.so                                   [.] 0x00000000006b6dc3
   1.06%  python   libdeepspeech.so                                   [.] 0x00000000006b6db3
   0.60%  python   libdeepspeech.so                                   [.] 0x00000000006b6d6e
   0.58%  python   libdeepspeech.so                                   [.] 0x00000000006b6dbb
```

Unfortunately, the relevant symbols are stripped from the binary! So
I'm not sure whether the neural network acoustic model or the n-gram
language model is the slow part. It's probably a safe bet that the
acoustic neural network is the bottleneck initially, though. We can
always speed up the n-gram language model by reducing the size of the
beam in beam search (which should be okay, with constrained grammars
defined by per-document language models).

### Using TPUs

First, a note: CPUs may be a reasonable platform for neural network
inference, if we allow.