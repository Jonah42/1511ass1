import csv

with open('debug/1600_bias.txt', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	print('[')
	for row in spamreader:
		text = ','.join(row)
		print(f'[{text[0:len(text)-2]}],')
	print('];')