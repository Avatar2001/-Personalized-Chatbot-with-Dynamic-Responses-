from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import json
from datasets import load_dataset, DatasetDict, Dataset


class DatasetProcessor(ABC):
    """Abstract base class for dataset processing operations."""
    
    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        """Process a dataset according to specific logic."""
        pass


class ColumnSelector(DatasetProcessor):
    """Selects specified columns from the dataset."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def process(self, dataset: Dataset) -> Dataset:
        available_columns = set(dataset.column_names)
        missing_columns = set(self.columns) - available_columns
        if missing_columns:
            raise ValueError(f"Columns not found in dataset: {missing_columns}")
        return dataset.select_columns(self.columns)


class FilterProcessor(DatasetProcessor):
    """Applies filtering logic to the dataset."""
    
    def __init__(self, filter_fn: Callable[[Dict[str, Any]], bool]):
        self.filter_fn = filter_fn
    
    def process(self, dataset: Dataset) -> Dataset:
        return dataset.filter(self.filter_fn)


class Formatter(DatasetProcessor):
    """Formats dataset into instruction-output pairs."""
    
    def __init__(self, instruction_template: str, output_column: str):
        """
        Args:
            instruction_template: Template string with column names in curly braces
            output_column: Name of the column to use as output
        """
        self.instruction_template = instruction_template
        self.output_column = output_column
    
    def process(self, dataset: Dataset) -> Dataset:
        def format_example(example):
            try:
                instruction = self.instruction_template.format(**example)
                return {
                    "instruction": instruction,
                    "output": example[self.output_column]
                }
            except KeyError as e:
                raise ValueError(f"Missing key in template: {e}")
        
        return dataset.map(format_example, remove_columns=dataset.column_names)


class HuggingFaceDatasetPipeline:
    """Main pipeline for processing Hugging Face datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        splits: Optional[List[str]] = None,
        processors: Optional[List[DatasetProcessor]] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            dataset_name: Hugging Face dataset identifier (e.g., "Azure99/stackoverflow-qa-top-300k")
            splits: List of splits to process (e.g., ["train", "test"]). If None, processes all splits.
            processors: List of processing steps to apply in order
        """
        self.dataset_name = dataset_name
        self.splits = splits
        self.processors = processors or []
    
    def add_processor(self, processor: DatasetProcessor) -> None:
        """Add a processing step to the pipeline."""
        self.processors.append(processor)
    
    def _process_split(self, dataset_split: Dataset) -> Dataset:
        """Apply all processors to a single dataset split."""
        processed = dataset_split
        for processor in self.processors:
            processed = processor.process(processed)
        return processed
    
    def run(self) -> DatasetDict:
        """Execute the full pipeline and return processed dataset."""
        # Load dataset
        raw_dataset = load_dataset(self.dataset_name)
        
        # Determine splits to process
        if self.splits is None:
            target_splits = list(raw_dataset.keys())
        else:
            missing_splits = set(self.splits) - set(raw_dataset.keys())
            if missing_splits:
                raise ValueError(f"Splits not found in dataset: {missing_splits}")
            target_splits = self.splits
        
        # Process each split
        processed_splits = {}
        for split in target_splits:
            print(f"Processing split: {split}")
            processed_splits[split] = self._process_split(raw_dataset[split])
        
        return DatasetDict(processed_splits)
    
    def save_to_json(self, output_path: str, split: Optional[str] = None) -> None:
        """
        Save processed dataset to JSON format.
        
        Args:
            output_path: Path to save JSON file
            split: Specific split to save (if None, saves first available split)
        """
        processed_data = self.run()
        
        if split is None:
            split = next(iter(processed_data.keys()))
        elif split not in processed_data:
            raise ValueError(f"Split '{split}' not found in processed data")
        
        # Convert to list of dictionaries
        data_list = [
            {"instruction": ex["instruction"], "output": ex["output"]}
            for ex in processed_data[split]
        ]
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data_list)} records to {output_path}")


# Convenience function for the specific Stack Overflow use case
def create_stackoverflow_pipeline(
    dataset_name: str = "Azure99/stackoverflow-qa-top-300k",
    splits: Optional[List[str]] = None
) -> HuggingFaceDatasetPipeline:
    """
    Create a pre-configured pipeline for Stack Overflow Python questions.
    
    Args:
        dataset_name: Hugging Face dataset identifier
        splits: Splits to process (e.g., ["train"])
    
    Returns:
        Configured HuggingFaceDatasetPipeline
    """
    pipeline = HuggingFaceDatasetPipeline(dataset_name, splits)
    
    # 1. Select required columns
    pipeline.add_processor(ColumnSelector(["title", "body", "answer_body", "tags"]))
    
    # 2. Filter for Python-related tags
    def python_filter(example):
        tags = example["tags"]
        return isinstance(tags, str) and "python" in tags.lower()
    
    pipeline.add_processor(FilterProcessor(python_filter))
    
    # 3. Format as instruction-output pairs
    pipeline.add_processor(Formatter(
        instruction_template="{title}, {body}",
        output_column="answer_body"
    ))
    
    return pipeline


# Example usage
if __name__ == "__main__":
    # Create and run the pipeline
    pipeline = create_stackoverflow_pipeline(splits=["train"])
    pipeline.save_to_json("stackoverflow_qa.json", split="train")
    
    # Alternative: Build custom pipeline
    # custom_pipeline = HuggingFaceDatasetPipeline("your/dataset", splits=["train"])
    # custom_pipeline.add_processor(ColumnSelector(["col1", "col2"]))
    # custom_pipeline.add_processor(FilterProcessor(lambda x: x["col1"] > 5))
    # custom_pipeline.add_processor(Formatter("{col1} details: {col2}", "col2"))
    # custom_pipeline.save_to_json("custom_output.json")