import progressbar


class ProgressBar:
    """
    Initializes the object.

    This method initializes the object by setting the `pbar` attribute to `None`.

    Parameters:
        self: The object instance.

    Returns:
        None
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
