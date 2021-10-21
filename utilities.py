import json
from urllib import request


def request_json_data(url):
  """
  Request url data
  :param url:
  :return:
  """
  with request.urlopen(url) as j:
    data = json.loads(j.read().decode())
  return data
