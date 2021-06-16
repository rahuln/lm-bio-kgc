"""
Code to get info on UMLS CUI's.

Based on https://github.com/HHS/uts-rest-api/blob/master/samples/python/retrieve-cui-or-code.py.
"""


from authentication import Authentication
import requests
import json


APIKEY = '' # ADD YOUR UMLS API KEY HERE
VERSION = "current"
AuthClient = Authentication(APIKEY)


def get_cui(identifier):
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/content/"+str(VERSION)+"/CUI/"+str(identifier)

    tgt = AuthClient.gettgt()

    # ticket is the only parameter needed for this call - paging does not come
    # into play because we're only asking for one Json object
    query = {'ticket': AuthClient.getst(tgt)}
    r = requests.get(uri + content_endpoint, params=query)
    r.encoding = 'utf-8'
    items = json.loads(r.text)
    jsonData = items["result"]

    definitions_endpoint = jsonData["definitions"]
    if definitions_endpoint == "NONE":
        jsonDefinitions = []
    else:
        tgt = AuthClient.gettgt()
        query = {'ticket': AuthClient.getst(tgt)}
        r = requests.get(definitions_endpoint, params=query)
        r.encoding = 'utf-8'
        definitions = json.loads(r.text)
        jsonDefinitions = definitions["result"]

    return {"data": jsonData,
            "definitions": jsonDefinitions}
