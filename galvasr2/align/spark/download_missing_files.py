# Copyright 2021 NVIDIA

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import internetarchive as ia
from pyspark import Row
import tqdm

# Quick script I wrote up to download missing rows discovered in
# align_lib_test.py It is probably better to just modify this script
# rather than try to make it robust for future use cases if you need
# to download more data in the future.


def download_missing_data(rows, file_name_key):
    for row in tqdm.tqdm(rows):
        ia.download(
            row.identifier,
            [row[file_name_key]],
            destdir="gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
            # Very important to set this. tf.io.gfile uses mtime in
            # nanoseconds, while archive.org uses mtime in seconds
            # (as far as I can tell). I could convert the
            # nanoseconds to seconds, of course, but don't want to
            # make an error.
            ignore_existing=True,
            # tf.io.gfile does not expose any functionality like os.utime
            no_change_timestamp=True,
            ignore_errors=False,
        )


def main():
    missing_audio_rows = [
        Row(
            identifier="gov.uscourts.ca3.14-14-00855-CV",
            audio_document_id="gov.uscourts.ca3.14-14-00855-CV.2010-09-21.mp3",
            text_document_id="gov.uscourts.ca3.14-14-00855-CV.2010-09-21.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca9.17-50255",
            audio_document_id="gov.uscourts.ca9.17-50255.2019-02-05.mp3",
            text_document_id="gov.uscourts.ca9.17-50255.2019-02-05.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="mnn_22274_1264",
            audio_document_id="22274_1264.mp3",
            text_document_id="22274_1264.asr.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.ca6.19-5836",
            audio_document_id="gov.uscourts.ca6.19-5836.2020-03-12.mp3",
            text_document_id="gov.uscourts.ca6.19-5836.2020-03-12.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca3.19-3305",
            audio_document_id="gov.uscourts.ca3.19-3305.2021-04-29.mp3",
            text_document_id="gov.uscourts.ca3.19-3305.2021-04-29.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca7.19-2985",
            audio_document_id="gov.uscourts.ca7.19-2985.2020-04-14.mp3",
            text_document_id="gov.uscourts.ca7.19-2985.2020-04-14.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca3.18-2038",
            audio_document_id="gov.uscourts.ca3.18-2038.2019-02-07.mp3",
            text_document_id="gov.uscourts.ca3.18-2038.2019-02-07.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.cadc.18-5356",
            audio_document_id="gov.uscourts.cadc.18-5356.2020-12-16.mp3",
            text_document_id="gov.uscourts.cadc.18-5356.2020-12-16.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.illappct.1-14-1960WC",
            audio_document_id="gov.uscourts.illappct.1-14-1960WC.2015-09-16.mp3",
            text_document_id="gov.uscourts.illappct.1-14-1960WC.2015-09-16.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.illappct.3-11-0083",
            audio_document_id="gov.uscourts.illappct.3-11-0083.2012-01-24.mp3",
            text_document_id="gov.uscourts.illappct.3-11-0083.2012-01-24.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.illappct.4-17-0227",
            audio_document_id="gov.uscourts.illappct.4-17-0227.2018-10-17.mp3",
            text_document_id="gov.uscourts.illappct.4-17-0227.2018-10-17.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca6.18-3720",
            audio_document_id="gov.uscourts.ca6.18-3720.2019-08-06.mp3",
            text_document_id="gov.uscourts.ca6.18-3720.2019-08-06.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca3.19-3390",
            audio_document_id="gov.uscourts.ca3.19-3390.2021-01-13.mp3",
            text_document_id="gov.uscourts.ca3.19-3390.2021-01-13.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca3.18-2490",
            audio_document_id="gov.uscourts.ca3.18-2490.2019-03-15.mp3",
            text_document_id="gov.uscourts.ca3.18-2490.2019-03-15.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="scm-443722-mmrtishighlightsfrom2014andpl",
            audio_document_id="ethioyouthmediatv_15_01_11.mp3",
            text_document_id="ethioyouthmediatv_15_01_11.asr.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.ca7.18-2517",
            audio_document_id="gov.uscourts.ca7.18-2517.2019-02-08.mp3",
            text_document_id="gov.uscourts.ca7.18-2517.2019-02-08.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="scm-200866-talkingstickdenismoynihanamyg",
            audio_document_id="talkingstick_13_01_23.mp3",
            text_document_id="talkingstick_13_01_23.asr.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.cafc.2018-2332",
            audio_document_id="gov.uscourts.cafc.2018-2332.2019-11-08.mp3",
            text_document_id="gov.uscourts.cafc.2018-2332.2019-11-08.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="om-400-132128",
            audio_document_id="JqaUkgx31tQ.mp3",
            text_document_id="132128_cuepoints.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.illappct.5-15-0384",
            audio_document_id="gov.uscourts.illappct.5-15-0384.2016-08-31.mp3",
            text_document_id="gov.uscourts.illappct.5-15-0384.2016-08-31.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca4.12-364-cv",
            audio_document_id="gov.uscourts.ca4.12-364-cv.2015-10-29.mp3",
            text_document_id="gov.uscourts.ca4.12-364-cv.2015-10-29.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="mnn_22274_1253",
            audio_document_id="22274_1253.mp3",
            text_document_id="22274_1253.asr.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.ca7.18-2187",
            audio_document_id="gov.uscourts.ca7.18-2187.2019-01-15.mp3",
            text_document_id="gov.uscourts.ca7.18-2187.2019-01-15.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="gov.uscourts.ca7.18-2571",
            audio_document_id="gov.uscourts.ca7.18-2571.2019-01-16.mp3",
            text_document_id="gov.uscourts.ca7.18-2571.2019-01-16.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
        Row(
            identifier="mnn_23078_388",
            audio_document_id="23078_388.mp3",
            text_document_id="23078_388.asr.srt",
            licenseurl="https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        Row(
            identifier="gov.uscourts.ca8.18-1312",
            audio_document_id="gov.uscourts.ca8.18-1312.2019-01-17.mp3",
            text_document_id="gov.uscourts.ca8.18-1312.2019-01-17.asr.srt",
            licenseurl="https://www.usa.gov/government-works",
        ),
    ]

    missing_audio_rows = [
        Row(
            identifier="gov.uscourts.ca4.12-364-cv",
            audio_document_id="gov.uscourts.ca4.12-364-cv.2015-10-29.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/house_proceeding_07-18-06_00_1.HQ.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/house_proceeding_07-19-06_1.HQ.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/senate_proceeding_02-27-07.HQ.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/senate_proceeding_03-06-06.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/senate_proceeding_03-08-06.mini.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/senate_proceeding_03-15-07.HQ.mp3",
        ),
        Row(
            identifier="metavid",
            audio_document_id="video_archive/senate_proceeding_05-05-06.mp3",
        ),
        Row(identifier="submedia_videos_2011", audio_document_id="AmateurRiot.mp3"),
    ]

    download_missing_data(missing_audio_rows, "audio_document_id")


if __name__ == "__main__":
    main()
