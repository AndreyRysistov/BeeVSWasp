import matplotlib.pyplot as plt

def plot_sample_images(x, y):
    fig = plt.figure(figsize=(12, 8))

    for image, cls, index in zip(x, y, range(9)):
        plt.subplot(3, 3, index + 1)
        plt.imshow(image)
        plt.title('Class: ' + str(cls))
        plt.axis('off')
    plt.show()