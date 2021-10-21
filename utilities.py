import json
from urllib import request
from html import unescape
import unicodedata


def request_json_data(url):
  """
  Request url data
  :param url:
  :return:
  """
  with request.urlopen(url) as j:
    data = json.loads(j.read().decode())
  return data


def clean_text(text):
  """
  Deal with artifacts and remove accent (e.g á -> a)
  :param text:
  :return:
  """
  text = unescape(text)
  return ''.join(c for c in unicodedata.normalize('NFD', text)
                 if unicodedata.category(c) != 'Mn')

