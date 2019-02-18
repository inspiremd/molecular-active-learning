import numpy as np

class Generator():
    def __init__(self, smiles):
        self.smiles = smiles
        self.indices = np.arange(0, smiles.shape[0], step=1)
        self.sampled = {}
        self.generator = self.create_generator(self.smiles, self.indices, self.sampled)

    def create_generator(self, smiles, indices, sampled):
        def generator():
            samples = np.random.choice(indices, size=smiles.shape[0], replace=False)
            for i in samples:
                if i in self.sampled and sampled[i]:
                    continue

                yield smiles[i]
                sampled[i] = True

        return generator



