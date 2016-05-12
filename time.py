import json
import sys

with open('sample2.json') as data_file:
    data = json.load(data_file)
    hand0 = data["hands"][0]["type"]
    for x in range(5):
        print(data["pointables"][x]["btipPosition"])
