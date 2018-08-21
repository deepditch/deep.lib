import unittest
import numpy as np
from validation import NHotAccuracy 

class Test(unittest.TestCase):
    def test_n_hot_accuracy(self):
        preds = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ])

        labels = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])

        acc = NHotAccuracy(4)
        acc.update_from_numpy(preds, labels)

        self.assertEqual(acc.count, 16)
        self.assertEqual(acc.num_correct, 8)

        self.assertEqual(acc.details[0]["correct_pos"], 2)
        self.assertEqual(acc.details[0]["correct_neg"], 0)
        self.assertEqual(acc.details[0]["false_pos"], 0)
        self.assertEqual(acc.details[0]["false_neg"], 2)

        self.assertEqual(acc.details[1]["correct_pos"], 0)
        self.assertEqual(acc.details[1]["correct_neg"], 2)
        self.assertEqual(acc.details[1]["false_pos"], 2)
        self.assertEqual(acc.details[1]["false_neg"], 0)

        self.assertEqual(acc.details[2]["correct_pos"], 2)
        self.assertEqual(acc.details[2]["correct_neg"], 0)
        self.assertEqual(acc.details[2]["false_pos"], 0)
        self.assertEqual(acc.details[2]["false_neg"], 2)

        self.assertEqual(acc.details[3]["correct_pos"], 0)
        self.assertEqual(acc.details[3]["correct_neg"], 2)
        self.assertEqual(acc.details[3]["false_pos"], 2)
        self.assertEqual(acc.details[3]["false_neg"], 0)


if __name__ == '__main__':
    unittest.main()