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

import cProfile
import os
import unittest

import pandas as pd

import dsalign_main
from galvasr2.align.spark.align_lib import fix_text_udf, srt_to_text
from galvasr2.align.spark.dsalign_lib import prepare_align_udf
from galvasr2.utils import find_runfiles

# Data created with:
# gsutil cp "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS/BOS2015April7/BOS 2015April7.asr.srt" galvasr2/align/spark/test_data/BOS2015April7/BOS\ 2015April7.asr.srt
# gsutil cp gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/decoder_ctm_dir/BOS2015April7-BOS_2015April7.mp3.ctm galvasr2/align/spark/test_data/BOS2015April7/BOS2015April7-BOS_2015April7.mp3.ctm

class DSAlignTest(unittest.TestCase):
    def test(self):
        dsalign_args = dsalign_main.parse_args("")
        alphabet_path = os.path.join(find_runfiles(), "__main__/galvasr2/align/spark/alphabet2.txt")
        align_udf = prepare_align_udf(dsalign_args, alphabet_path)
        align = align_udf.func
        transcript_names = pd.Series(["BOS2015April7/BOS 2015April7.asr.srt"])
        audio_names = pd.Series(["BOS2015April7/BOS 2015April7.mp3"])
        with open(os.path.join(find_runfiles(),
                               "__main__/galvasr2/align/spark/test_data/BOS2015April7/BOS2015April7.asr.srt"), "rb") as fh:
            transcript_series = srt_to_text.func(fix_text_udf.func(pd.Series([fh.read()])))
        with open(os.path.join(find_runfiles(),
                               "__main__/galvasr2/align/spark/test_data/BOS2015April7/BOS2015April7-BOS_2015April7.mp3.ctm"), "rb") as fh:
            ctm_content_series = fix_text_udf.func(pd.Series([fh.read()]))
        profiler = cProfile.Profile()
        profiler.enable()
        blah = align(transcript_names, audio_names, transcript_series, ctm_content_series)
        profiler.disable()
        profiler.print_stats(sort='time')
        print(blah)


#     name=BOS2015April7/BOS 2015April7.asr.srt audio_name=BOS2015April7/BOS 2015April7.mp3

# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BC20160712PlanningCommission/BC-2016-0712-PlanningCommission.asr.srt audio_name=BC20160712PlanningCommission/BC-2016-0712-PlanningCommission.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=December_2012_Holland_City_Connections_-_Public_Safety_Tour/December_2012_Holland_City_Connections_-_Public_Safety_Tour.asr.srt audio_name=December_2012_Holland_City_Connections_-_Public_Safety_Tour/December_2012_Holland_City_Connections_-_Public_Safety_Tour.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BulldogDrummondEscapes/BULLDOG_DRUMMOND_ESCAPES.asr.srt audio_name=BulldogDrummondEscapes/BULLDOG_DRUMMOND_ESCAPES.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOSFeb11/BOSFeb11.asr.srt audio_name=BOSFeb11/BOSFeb11.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BCPlanningCommission011216/BC_PlanningCommission_01-12-16.asr.srt audio_name=BCPlanningCommission011216/BC_PlanningCommission_01-12-16.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CC20160617RegularMeeting/CC2016-0617-RegularMeeting.asr.srt audio_name=CC20160617RegularMeeting/CC2016-0617-RegularMeeting.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BC20160209PlanningCommission/BC2016-0209-PlanningCommission.asr.srt audio_name=BC20160209PlanningCommission/BC2016-0209-PlanningCommission.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOSJune11/BOSJune11.asr.srt audio_name=BOSJune11/BOSJune11.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Digital_Marketing_Summer_Sessions_-_Advanced_Facebook_and_Facebook_Advertising/Digital_Marketing_Summer_Sessions_-_Advanced_Facebook_and_Facebook_Advertising.asr.srt audio_name=Digital_Marketing_Summer_Sessions_-_Advanced_Facebook_and_Facebook_Advertising/Digital_Marketing_Summer_Sessions_-_Advanced_Facebook_and_Facebook_Advertising.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CC20160802EAR/CC-2016-0802-EAR.asr.srt audio_name=CC20160802EAR/CC-2016-0802-EAR.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS2015March17/BOS2015March17.asr.srt audio_name=BOS2015March17/BOS2015March17.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BSEAC012914CL19/BSEAC_012914-CL19.asr.srt audio_name=BSEAC012914CL19/BSEAC_012914-CL19.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=APR2359_100304c/APR2359_100304c.asr.srt audio_name=APR2359_100304c/APR2359_100304c.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CC20160902RegMtg/CC-2016-0902-RegMtg.asr.srt audio_name=CC20160902RegMtg/CC-2016-0902-RegMtg.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=6909FoxboroughBoardOfSelectmen/2009_06_09BOS.asr.srt audio_name=6909FoxboroughBoardOfSelectmen/2009_06_09BOS.mp3
# logs/cuda_aligner_5n.log:^M[Stage 12:>                                                      (0 + 96) / 200]^MGALVEZ: timed out for name=BC20160324MolokaiPlanningCommissionPart2of2/BC-2016-0324-MolokaiPlanningCommission_part2of2.asr.srt audio_name=BC20160324MolokaiPlanningCommissionPart2of2/BC-2016-0324-MolokaiPlanningCommission_part2of2.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOSJan23/BOSJan23.asr.srt audio_name=BOSJan23/BOSJan23.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BC20160310MolokaiPlanningCommissionPart2of2/BC-2016-0310-MolokaiPlanningCommission_part2of2.asr.srt audio_name=BC20160310MolokaiPlanningCommissionPart2of2/BC-2016-0310-MolokaiPlanningCommission_part2of2.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS071116/BOS_071116.asr.srt audio_name=BOS071116/BOS_071116.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BC20161025PlanningCommSpecialMtg/BC-2016-1025-PlanningComm_SpecialMtg.asr.srt audio_name=BC20161025PlanningCommSpecialMtg/BC-2016-1025-PlanningComm_SpecialMtg.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BF41515/BF 4-15-15.asr.srt audio_name=BF41515/BF 4-15-15.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CCRegMtg151006/CC_RegMtg_151006.asr.srt audio_name=CCRegMtg151006/CC_RegMtg_151006.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=EAR12616/EAR 1-26-16.asr.srt audio_name=EAR12616/EAR 1-26-16.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS2016May17/BOS2016May17.asr.srt audio_name=BOS2016May17/BOS2016May17.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BC20160726PlanningCommission/BC-2016-0726-PlanningCommission.asr.srt audio_name=BC20160726PlanningCommission/BC-2016-0726-PlanningCommission.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS072715Title1/BOS_072715_title1.asr.srt audio_name=BOS072715Title1/BOS_072715_title1.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOSBMLD120808/BOS_BMLD_120808.asr.srt audio_name=BOSBMLD120808/BOS_BMLD_120808.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CCBF160328/CC-BF-160328.asr.srt audio_name=CCBF160328/CC-BF-160328.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Governance121015/Governance 12-10-15.asr.srt audio_name=Governance121015/Governance 12-10-15.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS091216CL19/BOS_091216-CL19.asr.srt audio_name=BOS091216CL19/BOS_091216-CL19.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Early_Bird_Breakfast_-_November/Early_Bird_Breakfast_-_November.asr.srt audio_name=Early_Bird_Breakfast_-_November/Early_Bird_Breakfast_-_November.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Board_of_Selectmen_-_December_15_2014./Board_of_Selectmen_-_December_15_2014..asr.srt audio_name=Board_of_Selectmen_-_December_15_2014./Board_of_Selectmen_-_December_15_2014..mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=52212FoxboroughBoardOfSelectmen/2012_05_22BOS.asr.srt audio_name=52212FoxboroughBoardOfSelectmen/2012_05_22BOS.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CC20160517EAR/CC-2016-0517-EAR.asr.srt audio_name=CC20160517EAR/CC-2016-0517-EAR.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS2015May26/BOS2015May26.asr.srt audio_name=BOS2015May26/BOS2015May26.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CCPlanning11192015/CC_Planning_11-19-2015.asr.srt audio_name=CCPlanning11192015/CC_Planning_11-19-2015.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS20130305/BOS2013-03-05.asr.srt audio_name=BOS20130305/BOS2013-03-05.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=31312FoxboroughBoardOfSelectmenMeeting/2012_03_13BOS.asr.srt audio_name=31312FoxboroughBoardOfSelectmenMeeting/2012_03_13BOS.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Conversations_with_Lady_Tea_5.16/Conversations_with_Lady_Tea_5.16.asr.srt audio_name=Conversations_with_Lady_Tea_5.16/Conversations_with_Lady_Tea_5.16.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=CCBF92815/CC BF 9-28-15.asr.srt audio_name=CCBF92815/CC BF 9-28-15.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS2015April7/BOS 2015April7.asr.srt audio_name=BOS2015April7/BOS 2015April7.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BOS111015MinutemanCL19/BOS_111015_Minuteman-CL19.asr.srt audio_name=BOS111015MinutemanCL19/BOS_111015_Minuteman-CL19.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=Beat_the_Devil_DVD/Beat.asr.srt audio_name=Beat_the_Devil_DVD/Beat.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=BHSSENIORAWARDS2016CL19/BHS SENIOR AWARDS 2016-CL19.asr.srt audio_name=BHSSENIORAWARDS2016CL19/BHS SENIOR AWARDS 2016-CL19.mp3
# logs/cuda_aligner_5n.log:GALVEZ: timed out for name=FinancialTaskForce103114CL10/Financial-Task-Force_103114-CL10.asr.srt audio_name=FinancialTaskForce103114CL10/Financial-Task-Force_103114-CL10.mp3

if __name__ == '__main__':
    unittest.main()
