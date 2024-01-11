import gzip
import io
import os
import subprocess
from collections import Counter


def convert_and_filter_topk(output_dir, input_txt, top_k):
    """Convert to lowercase, count word occurrences and save top-k words to a file"""

    counter = Counter()
    data_lower = output_dir + "." + "lower.txt.gz"

    print("\nConverting to lowercase and counting word occurrences ...")
    with io.TextIOWrapper(
        io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
    ) as file_out:
        # Open the input file either from input.txt or input.txt.gz
        _, file_extension = os.path.splitext(input_txt)
        if file_extension == ".gz":
            file_in = io.TextIOWrapper(
                io.BufferedReader(gzip.open(input_txt)), encoding="utf-8"
            )
        else:
            file_in = open(input_txt, encoding="utf-8")

        for line in file_in:
            line_lower = line.lower()
            counter.update(line_lower.split())
            file_out.write(line_lower)

        file_in.close()

    # Save top-k words
    print("\nSaving top {} words ...".format(top_k))
    top_counter = counter.most_common(top_k)
    vocab_str = "\n".join(word for word, count in top_counter)
    # race condition from writing to the same path?
    vocab_path = "vocab-{}.txt".format(top_k)
    vocab_path = output_dir + "." + vocab_path
    with open(vocab_path, "w+", encoding="utf-8") as file:
        file.write(vocab_str)

    print("\nCalculating word statistics ...")
    total_words = sum(counter.values())
    print("  Your text file has {} words in total".format(total_words))
    print("  It has {} unique words".format(len(counter)))
    top_words_sum = sum(count for word, count in top_counter)
    word_fraction = (top_words_sum / total_words) * 100
    print(
        "  Your top-{} words are {:.4f} percent of all words".format(
            top_k, word_fraction
        )
    )
    print('  Your most common word "{}" occurred {} times'.format(*top_counter[0]))
    last_word, last_count = top_counter[-1]
    print(
        '  The least common word in your top-k is "{}" with {} times'.format(
            last_word, last_count
        )
    )
    for i, (w, c) in enumerate(reversed(top_counter)):
        if c > last_count:
            print(
                '  The first word with {} occurrences is "{}" at place {}'.format(
                    c, w, len(top_counter) - 1 - i
                )
            )
            break

    return data_lower, vocab_str


def build_lm(
    output_dir,
    kenlm_bins,
    arpa_order,
    max_arpa_memory,
    arpa_prune,
    discount_fallback,
    binary_a_bits,
    binary_q_bits,
    binary_type,
    data_lower,
    vocab_str,
):
    print("\nCreating ARPA file ...")
    lm_path = output_dir + "." + "lm.arpa"
    subargs = [
        os.path.join(kenlm_bins, "lmplz"),
        "--order",
        str(arpa_order),
        "--temp_prefix",
        output_dir,
        "--memory",
        max_arpa_memory,
        "--text",
        data_lower,
        "--arpa",
        lm_path,
        "--prune",
        *arpa_prune.split("|"),
    ]
    if discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = output_dir + "." + "lm_filtered.arpa"
    subprocess.run(
        [
            os.path.join(kenlm_bins, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    # does it seriously quantize? wow!
    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    binary_path = output_dir + "." + "lm.binary"
    subprocess.check_call(
        [
            os.path.join(kenlm_bins, "build_binary"),
            "-s",
            "-a",
            str(binary_a_bits),
            "-q",
            str(binary_q_bits),
            "-v",
            binary_type,
            filtered_path,
            binary_path,
        ]
    )
