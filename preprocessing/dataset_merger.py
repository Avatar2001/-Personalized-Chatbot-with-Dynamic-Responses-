# preprocessing/dataset_merger.py
import random
from typing import List, Tuple

class DatasetMerger:
    def merge_and_shuffle(
        self,
        train_sets: List[List],
        val_sets: List[List],
        test_sets: List[List]
    ) -> Tuple[List, List, List]:
        merged_train = [item for subset in train_sets for item in subset]
        merged_val = [item for subset in val_sets for item in subset]
        merged_test = [item for subset in test_sets for item in subset]

        random.shuffle(merged_train)
        random.shuffle(merged_val)
        random.shuffle(merged_test)

        return merged_train, merged_val, merged_test
    
    def merge_lists(self, list_of_lists):
        return [item for sublist in list_of_lists for item in sublist if sublist]