import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
import tqdm


def create_ensemble_noisy_labels(model, dataloader, output_file, ensemble_epochs=1, tolerance=0.0, test_flag=False):
	for ensemble_epoch in tqdm.tqdm(range(ensemble_epochs)):
		outputs_per_ensemble = []
		all_paths = []
		# get all the outputs for each ensemble epoch
		for i, (images, paths) in enumerate(dataloader):
			if test_flag and i > 4:
				break
			images = images.cuda()
			with torch.no_grad():
				output = model(images)
			outputs_per_ensemble.append(output)
			all_paths.append(paths)
		outputs_per_ensemble = torch.cat(outputs_per_ensemble, dim=0)
		if ensemble_epoch == 0:
			all_outputs = outputs_per_ensemble
		else:
			all_outputs + outputs_per_ensemble

	print("This happened")
	all_outputs = all_outputs / ensemble_epochs
	# make a list of all paths
	all_paths1 = []
	for i in range(len(all_paths)):
		for j in range(len(all_paths[i])):
			all_paths1.append(all_paths[i][j])

	# from ensembled output, createon
	ensemble_output = np.array(all_outputs.cpu().numpy())
	bs = ensemble_output.shape[0]
	ensemble_output = torch.from_numpy(ensemble_output).float()
	y_preds = f.softmax(ensemble_output, dim=1)
	temp_pred = y_preds.to('cpu').numpy()
	#preds_max = np.max(temp_pred, axis=1)
	#preds_argmax = np.argmax(temp_pred, axis=1)

	# if prediction probability is higher that tolerance, append it final output
	confident_preds = 0
	final_result = []
	for num in range(bs):
		# if preds_max[num] >= tolerance:
		final_result.append([all_paths1[num], temp_pred[num]])
			# confident_preds += 1

	pd_sub = pd.DataFrame(final_result, columns=["file_name", "category_id"])
	pd_sub.to_csv(output_file, index=False)
