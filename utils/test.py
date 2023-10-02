import json 

classRGB = json.loads(open("../config/RGBColorDict.json").read())

# convert keys to list
className = list(classRGB.keys())
print(className)