count = 0
for i in range(10):
	for j in range(10):
		print(f"train data/digit/{j}_0{i}.pbm")
		count+=1
	if count == 10:
		print("validate")
		count = 0
for j in range(10,80):
	for i in range(10):
		print(f"train data/digit/{i}_{j}.pbm")
		count+=1
	if count == 10:
		print("validate")
		count = 0
# for i in range(10):
# 	print(f"run data/digit/{i}_00.pbm")
print("validate")
print("quit")