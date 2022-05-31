import re
def srt_to_text(path:str):
    final_text = ''
    for word in text.split():
        if re.match('\D+[^.]+[^-->]', word ):
            final_text = final_text + ' ' + word
    return final_text

clean_timestamps = open("greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety-gold.txt", "r")
text = clean_timestamps.read()
gold_srt = srt_to_text(text)


f = open("greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety-gold2.txt", "a")
f.write(gold_srt)