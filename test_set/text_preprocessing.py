import re

import num2words

def replace_ordinal_numbers(text):
    re_results = re.findall('(\d+(st|nd|rd|th))', text)
    for entire_result, suffix in re_results:
        n = int(entire_result[:-len(suffix)])
        text = text.replace(entire_result, num2words.num2words(n, ordinal=True))
    return text

def clean_text(text):
    # Remove text enclosed in parenthesis
    normalized_text = re.sub("[\(\[].*?[\)\]]", "", text)
    # Spell out ordinal numbers
    normalized_text = replace_ordinal_numbers(normalized_text)
    # Remove special characters except apostrophes
    normalized_text = re.sub("\n+", " ", normalized_text)
    normalized_text = re.sub("[^A-Za-z0-9 ']+", "", normalized_text)
    normalized_text = re.sub(" +", " ", normalized_text)

    normalized_text = normalized_text.lower()
    normalized_text = normalized_text.strip()
    return normalized_text
    