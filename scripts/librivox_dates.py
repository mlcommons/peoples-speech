from collections import defaultdict
from datetime import date
import json
from tqdm import tqdm
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import requests

END_POINT = "https://librivox.org/api/feed/audiobooks"

def get_librivox_urls() -> List:
  offset = 0
  limit = 1000
  results = []
  while True:
    print(offset)
    query = f"{END_POINT}/?format=json&fields[]=id&fields[]=totaltimesecs&fields[]=url_librivox&limit={limit}&offset={offset}"
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
  if len(matches) != 1:
    print("Suspicious url:", result["url_librivox"])
    return None
  date_text = matches[0].fetchNextSiblings()[0].contents[0]
  return date.fromisoformat(date_text)


def main():
  # results = get_librivox_urls()
  # with open("api_queries.json", "w") as fh:
  #   json.dump(results, fh)
  with open("api_queries_fixed.json", "r") as fh:
    results = json.load(fh)
  year_to_publications = defaultdict(lambda: 0)
  for result in tqdm(results):
    date = get_publication_datetime(result)
    if date is not None:
      year_to_publications[date.year] += result["totaltimesecs"]
  print(year_to_publications)

def graph(): # year_to_publications):
  year_to_publications = {2007: 13440999, 2008: 18323820, 2006: 4440603, 2009: 23704545, 2012: 23828004, 2005: 328280, 2016: 21511661, 2011: 24978573, 2010: 24405905, 2014: 21597865, 2015: 21960293, 2013: 24856543, 2018: 21273952, 2017: 22565206, 2019: 24020607, 2020: 12030849.219999999}
  keys, values = zip(*[(k, v) for (k, v) in year_to_publications.items()])
  df = pd.DataFrame({"year": keys, "seconds": values})
  df.sort_values(by=['year'], inplace=True, ascending=True)
  df["cumulative_hours"] = df.seconds.cumsum() /  60 / 60
  print(df)
  ax = df.plot.bar(x="year", y="cumulative_hours", rot=0)
  plt.show()

if __name__ == '__main__':
  # url_data = get_librivox_urls()
  # with open("api_queries_fixed.json", "w") as fh:
  #   json.dump(url_data, fh)
  graph()
  # main()
