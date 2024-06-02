import matplotlib.pyplot as plt

def format_duration(duration):
    hour = duration // (60 * 60)
    min = (duration % (60 * 60)) // 60
    second = duration % 60
    return f"{hour}: {min}: {second}"

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot(generated_images, ground_truth, condition, steps, save=False, save_path=""):
    nrows = ground_truth.shape[0] #batch size
    ncols = len(steps) + 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows), squeeze=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    for row in range(nrows):
        for col, t in enumerate(steps):
            ax = axes[row, col]
            img = (generated_images[t][row] + 1) / 2 # Convert from [-1, 1] to [0, 1]
            ax.imshow(img.permute(1,2,0))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Plot Condition Image - synthetic low light image
        ax = axes[row, ncols - 2]
        img = (condition[row].cpu() + 1) / 2 # Convert from [-1, 1] to [0, 1]
        ax.imshow(img.permute(1,2,0))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Plot Ground truth image
        ax = axes[row, ncols - 1]
        img = (ground_truth[row].cpu() + 1) / 2 # Convert from [-1, 1] to [0, 1]
        ax.imshow(img.permute(1,2,0))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    # Save plot
    if save:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.tight_layout()
