import sys


count = 0
total = 0
with open(sys.argv[1],'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(", ")
        total += float(line[-1].rstrip())
        count += 1

print("Avg for {} file is {}".format(sys.argv[1],total/count))