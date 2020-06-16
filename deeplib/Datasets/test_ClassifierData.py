import unittest
import ClassifierData
import numpy as np

class Test(unittest.TestCase):
    def test_make_n_hot_labels(self):
        test_labels = [
            ['cat', 'ant', 'dog'],
            ['plane', 'ant'],
            ['cat'],
        ]

        n_hot, classes = ClassifierData.make_n_hot_labels(test_labels)
        self.assertEqual(classes, ['ant', 'cat', 'dog', 'plane'])
        self.assertTrue(np.array_equal(n_hot, [[1,1,1,0], [1,0,0,1], [0,1,0,0]]))

if __name__ == '__main__':
    unittest.main()