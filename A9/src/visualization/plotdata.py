import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path
from pathlib import Path


class PlotData:

    def showImagesfromdataset(dataiterator, classes):
        images, labels = next(dataiterator)
        images = images.numpy()  # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display 20 images
        for idx in np.arange(20):
            ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
            from src.utils import utils
            utils.Utils.imshow(images[idx])
            ax.set_title(classes[labels[idx]])

        plt.savefig(Path(os.getcwd()).as_posix() + "/images/imagesfromdataset.png")

    def plotmisclassifiedimages(dataiterator, model, classes):
        images, labels = next(dataiterator)
        images.numpy()

        # move model inputs to cuda
        images = images.cuda()

        # get sample outputs
        output = model(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.cpu().numpy())

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(15, 20))

        loc = 0
        for idx in np.arange(128):
            if preds[idx] != labels[idx].item() and loc < 25:
                ax = fig.add_subplot(5, 5, loc + 1, xticks=[], yticks=[])
                from src.utils import utils
                utils.Utils.imshow(images[idx].cpu())
                ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]])
                             , color="red")
                loc += 1

        plt.savefig(Path(os.getcwd()).as_posix()+"/images/missclassifiedimages.png")

    def plottesttraingraph(train_losses, train_acc, test_losses, test_acc, lr_data):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
        axs[2, 0].plot(lr_data)
        axs[2, 0].set_title("Learning Rate")
        plt.savefig(Path(os.getcwd()).as_posix() + "/images/traintestgraphs.png")
        plt.show()

