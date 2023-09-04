import numpy as np
import matplotlib.pyplot as plt


# Step 1: Read data from the file and extract episode numbers and delay values
data = open("ddpg_delay.txt", "r").readlines()

# Step 3: Create lists for episode numbers and delay values
episodes = []
delays = []
for line in data:
    if line.startswith("episodes"):
        episode = float(line.split(":")[1])
        episodes.append(episode)
    elif line.startswith("delay"):
        delay = float(line.split(":")[1])
        delays.append(delay)
# Step 4: Plot the graph

plt.plot(episodes, delays, marker='o')

# Step 5: Customize the plot
# plt.xlim(1, 5)  # Set the limits for the x-axis
# plt.ylim(1, 10)  # Set the limits for the y-axis

# Step 6: Add labels and title
plt.xlabel("Episode")
plt.ylabel("Delay")
plt.title("Delay vs. Episode")

# Step 7: Display the plot
plt.show()
