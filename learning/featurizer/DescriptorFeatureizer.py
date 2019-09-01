from mordred import Calculator, descriptors


class DescriptorFeaturizer(object):
    def __init__(self, **kwargs):
        """
        Featureizer for usign chemical descriptors
        :param kwargs: keyword arugments for MOrdered Calculator
        """
        self.calc = Calculator(descriptors, **kwargs)

    def __call__(self, rdkit_mol):
        """
        Returns descriptors for a given single molecule
        :param rdkit_mol: valid RDKIT molecule
        :return: numpy array with descriptors.
        """
        des = self.calc(rdkit_mol)

        return des
