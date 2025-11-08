# preprocessing/data_splitter.py
from typing import List, Tuple, Any
from sklearn.model_selection import train_test_split

class DataSplitter:
    """
    Splits a dataset into train, validation, and test sets.
    
    Follows a two-stage split:
      1. Split off `test_size` for (val + test)
      2. Split that portion into val and test using `val_ratio_of_test`
    
    Example:
        total = 1000, test_size=0.15 → 850 train, 150 (val+test)
        val_ratio_of_test=0.48 → val = 72, test = 78
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split(
        self,
        data: List[Any],
        test_size: float = 0.15,
        val_ratio_of_test: float = 0.48
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: List of records to split.
            test_size: Proportion of data to reserve for (val + test).
            val_ratio_of_test: Proportion of the test portion to use for validation.
        
        Returns:
            Tuple of (train, validation, test) lists.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if not (0 <= val_ratio_of_test <= 1):
            raise ValueError("val_ratio_of_test must be between 0 and 1")

        # First split: train vs (val + test)
        train_data, val_test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=self.random_state
        )

        # Second split: val vs test
        if len(val_test_data) == 0:
            return train_data, [], []

        val_data, test_data = train_test_split(
            val_test_data,
            test_size=val_ratio_of_test,
            random_state=self.random_state
        )

        return train_data, val_data, test_data