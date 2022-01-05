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
			epsilon = str(dict_line["epsilon"])
			avg_pck_threshold = str(dict_line["avg_pck_threshold"])
			total_time_avg_pck_threshold = str(dict_line["total_time_avg_pck_threshold"])

			if epsilon == "0.02":
				if (k := (n_drones, epsilon, avg_pck_threshold, total_time_avg_pck_threshold)) not in data:
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
	for (n_drone, epsilon, avg_pck, total_time_avg_pck), score in data.items():
		if not n_drone in n_drones and n_drone != "5":
			n_drones.append(n_drone)

		if n_drone not in dict_min_value_fe_drone:
			dict_min_value_fe_drone[n_drone] = (score, (epsilon, avg_pck, total_time_avg_pck))
		else:
			if dict_min_value_fe_drone[n_drone][0] > score:
				dict_min_value_fe_drone[n_drone] = (score, (epsilon, avg_pck, total_time_avg_pck))

		if (k := (epsilon, avg_pck, total_time_avg_pck)) not in dict_fe_tuning:
			dict_fe_tuning[k] = dict()

		dict_fe_tuning[(epsilon, avg_pck, total_time_avg_pck)].update({n_drone: score})

	colors = {v:k for k, v in cols.get_named_colors_mapping().items()}
	for (e, ap, ttap), v in dict_fe_tuning.items():
		scores = [v[x] for x in n_drones]
		p, = plt.plot(n_drones, scores, label=f"epsilon={e}, avg_pckt={ap}, tot_time_avg_pckt={ttap}")
		print(f"{colors[p.get_color()]} - (epsilon={e}, avg_pckt={ap}, tot_time_avg_pckt={ttap})")

	plt.xlabel("Number of drones")
	plt.ylabel("Scores")
	plt.legend(loc="upper right")
	plt.show()

	return dict_min_value_fe_drone


if __name__ == "__main__":
	data = load_data(file=os.path.join("data", "metrics.json"))
	min_values = plot(data)
	print(min_values)