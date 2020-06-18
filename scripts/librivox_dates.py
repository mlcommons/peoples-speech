from collections import defaultdict
from datetime import date
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
import requests

END_POINT = "https://librivox.org/api/feed/audiobooks"

def get_librivox_urls() -> List:
  offset = 0
  limit = 1000
  results = []
  while True:
    print(offset)
    query = f"{END_POINT}/?format=json&fields[]=id&fields[]=totaltimesecs&fields[]=url_librivox&limit=100&offset={offset}"
    offset += limit
    try:
      results.extend(requests.get(query).json()["books"])
    except KeyError:
      # Will return an empty dictionary when offset is too large
      break

  return results


def get_publication_datetime(result: Dict) -> Optional[date]:
  try:
    html_text = requests.get(result["url_librivox"]).text
  except requests.exceptions.MissingSchema as e:
    if "Invalid URL" in str(e):
      return None
  parsed = BeautifulSoup(html_text, 'html.parser')
  matches = [dt_tag for dt_tag in parsed.find_all("dt") if "Catalog date:" in dt_tag]
  assert len(matches) == 1
  date_text = matches[0].fetchNextSiblings()[0].contents[0]
  return date.fromisoformat(date_text)


def main():
  # results = get_librivox_urls()
  # with open("api_queries.json", "w") as fh:
  #   json.dump(results, fh)
  with  open("api_queries.json", "r") as fh:
    results = json.load(fh)
  year_to_publications = defaultdict(lambda: 0)
  for result in tqdm(results):
    date = get_publication_datetime(result)
    if date is not None:
      year_to_publications[date.year] += result["totaltimesecs"]
      print(sum(year_to_publications.values()))
  print(year_to_publications)

def graph(): # year_to_publications):
  year_to_publications = {2007: 2489768, 2008: 2667738, 2006: 931954, 2009: 2055170, 2012: 2330681, 2005: 129590, 2010: 2424765, 2011: 2697939, 2015: 2101654, 2017: 1691746, 2013: 2051202, 2014: 1973722, 2016: 1663480, 2020: 1137318, 2018: 1686594, 2019: 2160708}
  print([(k, v) for (k, v) in year_to_publications.items()])
  keys, values = zip(*[(k, v) for (k, v) in year_to_publications.items()])
  df = pd.DataFrame({"year": keys, "seconds": values})
  df.sort_values(by=['year'], inplace=True, ascending=True)
  df["cumulative_hours"] = df.seconds.cumsum() /  60 / 60
  print(df)
  # dates = matplotlib.dates.date2num(list_of_datetimes)


if __name__ == '__main__':
  graph()
