#!/usr/bin/env python

import srt
import sys

print(" ".join(line.content.replace("\n", " ") for line in srt.parse(sys.stdin)))
