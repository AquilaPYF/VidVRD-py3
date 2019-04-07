import json

with open('test.json', 'r') as in_f:
    data = json.load(in_f)

data.sort(key=lambda a: a['score'], reverse=True)
for each_ins in data:
    print(each_ins['triplet'], each_ins['duration'], each_ins['score'])
