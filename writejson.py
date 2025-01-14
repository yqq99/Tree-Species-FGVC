import os
import json

files = os.listdir('E:/PaperData/Plant/plant_2015_type/Leaf/')
json_file = 'E:/PaperData/Plant/PlantCLEF2015TrainingData/class_indices_v1.json'
json_file = 'class_indices_leaf.json'
no = 0

data = {}

for i in files:
    #data[i] = no;
    data[no] = i
    no += 1

with open(json_file, 'a', encoding='utf-8') as fw:
    str = json.dumps(data, indent=4, ensure_ascii=False)
    fw.write(str)
    fw.write('\n')
