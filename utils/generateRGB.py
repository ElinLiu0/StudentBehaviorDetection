import yaml
import json
# Read the YAML file
with open("../config/data.yaml",'r') as stream:
    data = yaml.load(stream.read(),Loader=yaml.Loader)
    names = data['names']
    print(names)

import random

RGBColorDict = {}
for i in names:
    RGBColorDict[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
with open("../config/RGBColorDict.json",'w') as stream:
    json.dump(RGBColorDict, stream)