# Running the test

To run the test locally, gentle receives a .txt file without timestamps and an audio file.

Setup your directory with a copy of the script_text directory, containing the .srt and .flac file. 

1. Run 
    `cd $(pwd)/script_text`
2. Run 
`python ts_remove.py`
3. Run
`cd ..`
4. Run 
`docker run -w /gentle --volume $(pwd)/script_test:/gentle/input lowerquality/gentle python align.py input/greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety.flac input/greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety-gold2.txt > transcript.json`