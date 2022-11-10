import matplotlib.pyplot as plt
import numpy as np


def visualize_augmentations(dataset, samples=10, cols=5):
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    random_idx = np.random.randint(0, len(dataset), size=samples)
    for i, idx in enumerate(random_idx):
        image, label = dataset[idx]
        ax.ravel()[i].imshow(np.squeeze(image).T, cmap="gray")
        ax.ravel()[i].set_title(dataset.classes[label])
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
