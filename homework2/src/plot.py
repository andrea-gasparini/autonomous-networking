import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import json
import os


def load_data(file: str):
	data = dict()
	with open(file, mode="r") as stream:
		while line := stream.readline():
			dict_line = json.loads(line)
			n_drones = str(dict_line["n-drones"])
			alpha = str(dict_line["alpha"])
			epsilon = str(dict_line["epsilon"])
			gamma = str(dict_line["gamma"])

			if (k := (n_drones, alpha, epsilon, gamma)) not in data:
				data[k] = []

			data[k].append((dict_line["score"], dict_line["seed"]))

	for k, value in data.items():
		data[k] = sum(list(map(lambda x: x[0], data[k]))) / len(data[k])

	return data


def plot(data) -> None:

	# Get number of drones and minimum value and construct dict for each n drone
	n_drones = []
	dict_min_value_fe_drone = {}
	dict_fe_tuning = dict()
	for (n_drone, alpha, epsilon, gamma), score in data.items():
		if not n_drone in n_drones:
			n_drones.append(n_drone)

		if n_drone not in dict_min_value_fe_drone:
			dict_min_value_fe_drone[n_drone] = (score, (alpha, epsilon, gamma))
		else:
			if dict_min_value_fe_drone[n_drone][0] > score:
				dict_min_value_fe_drone[n_drone] = (score, (alpha, epsilon, gamma))

		if (k := (alpha, epsilon, gamma)) not in dict_fe_tuning:
			dict_fe_tuning[k] = dict()

		dict_fe_tuning[(alpha, epsilon, gamma)].update({n_drone: score})

	colors = {v:k for k, v in cols.get_named_colors_mapping().items()}
	for (a, e, g), v in dict_fe_tuning.items():
		scores = [v[x] for x in n_drones]
		p, = plt.plot(n_drones, scores, label=f"alpha={a}, epsilon={e}, gamma={g}")
		print(f"{colors[p.get_color()]} - (alpha={a}, epsilon={e}, gamma={g})")

	plt.xlabel("Number of drones")
	plt.ylabel("Scores")
	plt.legend(loc="upper right")
	plt.show()

	return dict_min_value_fe_drone


if __name__ == "__main__":
	data = load_data(file=os.path.join("data", "metrics.json"))
	min_values = plot(data)
	print(min_values)