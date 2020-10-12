import sys

import lingvo.compat as tf
from lingvo.core import py_utils
from lingvo.core.ops import ascii_to_token_id
from lingvo.core.ops import id_to_ascii
from lingvo.core.ops import str_to_vocab_tokens
from lingvo.core.ops import vocab_id_to_token


tf.flags.DEFINE_string('in_words_txt', None,
                       'Name of input file with each word in vocabulary, new-line delimited.')
tf.flags.DEFINE_string('in_units_txt', None,
                       'Name of input file with each character in vocabulary, new-line delimited.')
tf.flags.DEFINE_string('out_spelling_txt', None,
                       'Name of output file. Will be in Kaldi\'s lexicon.txt format')
tf.flags.DEFINE_string('out_spelling_numbers_txt', None,
                       'Name of output file. Will be in Kaldi\'s lexicon_numbers.txt format')
tf.flags.DEFINE_string('out_units_txt', None,
                       'Name of output file. Will be in Kaldi\'s units.txt format')
tf.flags.DEFINE_string('space_char', None,
                       'Space charactr. " " is invalid for openfst.')

FLAGS = tf.flags.FLAGS

UNK_NUMBER = None

def dump_units_txt(in_units_txt: str, out_units_txt: str):
  with open(in_units_txt, "r") as in_fh, open(out_units_txt, "w") as out_fh:
    seen_unk = False
    seen_space = False
    for i, line in enumerate(in_fh):
      line = line.rstrip("\n")
      if line == "<unk>":
        seen_unk = True
        global UNK_NUMBER
        UNK_NUMBER = i
      if line == " ":
        line = FLAGS.space_char
        seen_space = True
      out_fh.write(f"{line} {i}\n")
    assert seen_unk
    assert seen_space
    
def main(unused_argv):
  dump_units_txt(FLAGS.in_units_txt, FLAGS.out_units_txt)
  dump_spellings()

def dump_spellings():
  words = []
  with open(FLAGS.in_words_txt, 'r') as words_fh:
    words = words_fh.read().lower().splitlines()
  # if "<unk>" not in words:
  #   words.append("<unk>")
  # We add 2 to account for <s> and (optional) </s> tokens.
  longest_word_length = max(len(word) for word in words) + 2

  print("GALV:", longest_word_length)

  with open(FLAGS.in_units_txt, 'r') as units_fh:
    vocab_tokens = [line.rstrip("\n") for line in units_fh.readlines()]

  print("GALV:", vocab_tokens)

  @tf.function(input_signature=[tf.TensorSpec(shape=[len(words)], dtype=tf.string)])
  def tokenize_words(words_t):
    padded_tokenized_t, _, paddings_t = str_to_vocab_tokens(
        labels=words_t,
        maxlen=longest_word_length,
        append_eos=True,
        pad_to_maxlen=True,
        vocab_filepath=FLAGS.in_units_txt,
        load_token_ids_from_vocab=False,
        delimiter=''
    )
    # Either lengths or paddings are incorrect.
    lengths_t = py_utils.LengthsFromPaddings(paddings_t)
    ragged_tokenized_t = tf.RaggedTensor.from_tensor(padded_tokenized_t, lengths=lengths_t)
    # Drop start-of-sentence-token
    ragged_tokenized_t = ragged_tokenized_t[:, 1:]
    lengths_t -=  1
    letters_t = vocab_id_to_token(id=ragged_tokenized_t.flat_values,
                                  vocab=vocab_tokens,
                                  load_token_ids_from_vocab=False)
    ragged_letters_t = tf.RaggedTensor.from_row_lengths(letters_t, lengths_t)
    # Is capatilizationt he problem?
    return ragged_tokenized_t, ragged_letters_t

  with tf.Session() as session:
    spelling_numbers, spelling_letters = session.run(tokenize_words(words))
  spelling_numbers = spelling_numbers.to_list()
  spelling_letters = spelling_letters.to_list()

  with open(FLAGS.out_spelling_txt, "w") as spelling_fh, open(FLAGS.out_spelling_numbers_txt, "w") as spelling_numbers_fh:
    for word, numbers, letters in zip(words, spelling_numbers, spelling_letters):
      if isinstance(letters, list):
        letters_str = " ".join([str(letter) for letter in word])
      else:
        letters_str = letters
      numbers_str = " ".join([str(number) for number in numbers])
      spelling_fh.write(f"{word} {letters_str}\n")
      spelling_numbers_fh.write(f"{word} {numbers_str}\n")
    spelling_fh.write("<unk> <unk>\n")
    spelling_numbers_fh.write(f"<unk> {UNK_NUMBER}\n")

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('in_words_txt')
  tf.flags.mark_flag_as_required('in_units_txt')
  tf.flags.mark_flag_as_required('out_spelling_txt')
  tf.flags.mark_flag_as_required('out_spelling_numbers_txt')
  tf.flags.mark_flag_as_required('out_units_txt')
  tf.flags.mark_flag_as_required('space_char')
  
  FLAGS(sys.argv)
  tf.app.run(main)
