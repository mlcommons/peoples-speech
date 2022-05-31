## Running the test
To run the test locally, gentle receives a .txt file without timestamps and an audio file.

Clone the repo and do the following:

1. Run `cd peoples-speech && git checkout rafael/aligner_test && cd script_test`

2. Run `python ts_remove.py && cd ..`

3. Run 
    docker run -w /gentle --volume $(pwd)/script_test:/gentle/input lowerquality/gentle python align.py input/greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety.flac input/greg-cleaned-test-set_170511BCFirePublicSafety_170511-BC-FirePublicSafety-gold2.txt > transcript.json