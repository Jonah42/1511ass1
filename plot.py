#Make a pretty picture
#import sys
import matplotlib.pyplot as plt

data = []
index = []
i = 0
nodes = input()
text = input()
while text != "done":
	data.append(float(text))
	index.append(i)
	i+=1
	text = input()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(index, data)
ax.set(title = "The 5 minute journal", xlabel = "Epoch", ylabel = "Accuracy")
plt.savefig(f"results/Nodes/{nodes}.png")
# ax.plot(data)
# plt.show()