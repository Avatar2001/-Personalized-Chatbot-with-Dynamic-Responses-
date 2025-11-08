# main.py    
global_counter = 0

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from preprocessing.config_loader import ConfigLoader
from preprocessing.logger_factory import setup_logger
from preprocessing.data_loader import DatasetLoader
from preprocessing.text_cleaner import TextCleaner
from preprocessing.formatters.dialogue_formatter import DialogueFormatter
from preprocessing.formatters.instruction_formatter import InstructionFormatter
from preprocessing.data_splitter import DataSplitter
from preprocessing.dataset_merger import DatasetMerger
from preprocessing.deduplicator import Deduplicator
from preprocessing.difficulty_merger import DifficultyMerger
from preprocessing.model_formatters.dialogue_model_formatter import DialogueModelFormatter
from preprocessing.issue_scanner import IssueScanner

def main():
    config = ConfigLoader()
    log_config = config.get("logging")
    output_dir = config.get("output.base_dir")
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, log_config["log_file"])
    logger = setup_logger(
        level=log_config["level"],
        log_format=log_config["format"],
        log_file=log_file,
        console=False
    )

    # Initialize services
    loader = DatasetLoader()
    cleaner = TextCleaner(logger=logger)
    dialogue_formatter = DialogueFormatter(logger=logger)
    instruction_formatter = InstructionFormatter(logger=logger)
    splitter = DataSplitter(random_state= config.get("splitting.random_state"))
    merger = DatasetMerger()
    deduplicator = Deduplicator(logger=logger)
    difficulty_merger = DifficultyMerger(logger=logger)
    model_formatter = DialogueModelFormatter(logger=logger)
    scanner = IssueScanner(logger=logger)

    # === Process Dialogue Datasets ===
    logger.info(" Starting preprocessing pipeline...")

    # === PersonaChat ===
    persona_cfg = config.get("datasets.persona-chat")
    logger.info(f"Loading {persona_cfg['name']}")
    persona_ds = loader.load_huggingface_dataset(persona_cfg["name"])

    # --- Scanning Before Cleaning (PersonaChat) ---
    logger.info("Scanning PersonaChat BEFORE cleaning...")
    verbose_scan = config.get("scanning.verbose", False)
    issues_before = scanner.scan_by_dataset_name(
        data_source="Cynaptics/persona-chat",
        dataset=persona_ds,
        subset="train",
        verbose= verbose_scan
    )
    logger.info(f"Issues before cleaning: {issues_before}")
    
    def clean_persona(ex):
        global global_counter
        if(global_counter < 0):
            print(ex)
        else:
            global_counter += 1
        ex["dialogue"] = cleaner.clean_dialogue_text(ex["dialogue"])
        return ex

    persona_ds["train"] = persona_ds["train"].map(clean_persona)

    # --- Scanning After cleaning (PersonaChat)---
    logger.info("Scanning PersonaChat AFTER cleaning...")
    issues_after = scanner.scan_by_dataset_name(
        data_source="Cynaptics/persona-chat",
        dataset=persona_ds,
        subset="train",
        verbose=verbose_scan
    )
    logger.info(f"Issues after cleaning: {issues_after}")

    logger.info("Formatting PersonaChat")
    processed_persona = dialogue_formatter.format_records(persona_cfg, persona_ds, "train")
    train_p, val_p, test_p = splitter.split(
        processed_persona,
        test_size= config.get("splitting.test_size"),
        val_ratio_of_test= config.get("splitting.val_ratio_of_test")
    )

    pfx = config.get("output.file_prefixes")
    for name, data in [("train", train_p), ("validation", val_p), ("test", test_p)]:
        path = os.path.join(output_dir, f"{pfx['persona']}_{name}.json")
        loader.save_json(data, path)
        logger.info(f"Saved {name}: {len(data)} dialogues → {path}")

    # === DailyDialog ===
    daily_cfg = config.get("datasets.daily-dialog")
    logger.info(f"Loading {daily_cfg['name']}")
    daily_ds = loader.load_huggingface_dataset(daily_cfg["name"])
    results = {}

    # Clean the data
    def clean_daily(ex):
        #ex["utterance"] = cleaner.clean_dialogue_text([ex["utterance"]])[0]
        return {"utterance": cleaner.clean_dialogue_text([ex["utterance"]])[0]}
    
    for subset in daily_cfg["subsets"]:
        # Scan BEFORE cleaning
        logger.info(f"[SCAN] Scanning DailyDialog '{subset}' BEFORE cleaning...")
        issues_before = scanner.scan_by_dataset_name(
            data_source=daily_cfg["name"],
            dataset=daily_ds,
            subset=subset,
            verbose=verbose_scan
        )
        logger.info(f"[SCAN] Issues before cleaning ({subset}): {issues_before}")

        # Apply Cleaning
        daily_ds[subset] = daily_ds[subset].map(clean_daily)

        # Scan AFTER cleaning
        logger.info(f"[SCAN] Scanning DailyDialog '{subset}' AFTER cleaning...")
        issues_after = scanner.scan_by_dataset_name(
            data_source=daily_cfg["name"],
            dataset=daily_ds,
            subset=subset,
            verbose=verbose_scan
        )
        logger.info(f"[SCAN] Issues after cleaning ({subset}): {issues_after}")

        # Format and save
        processed = dialogue_formatter.format_records(daily_cfg, daily_ds, subset)
        results[subset] = processed

        path = os.path.join(output_dir, f"{pfx['daily']}_{subset}.json")
        loader.save_json(processed, path)
        logger.info(f"Saved {subset}: {len(processed)} dialogues → {path}")

    # === Merge ===
    logger.info("Merging datasets")
    merged_train, merged_val, merged_test = merger.merge_and_shuffle(
        [train_p, results["train"]],
        [val_p, results["validation"]],
        [test_p, results["test"]]
    )

    for name, data in [("train", merged_train), ("validation", merged_val), ("test", merged_test)]:
        path = os.path.join(output_dir, f"{pfx['merged']}_{name}.json")
        loader.save_json(data, path)
        logger.info(f"Saved merged {name}: {len(data)} dialogues → {path}")

    # === Format for Model ===
    logger.info("Formatting for model input")
    for name, data in [("train", merged_train), ("validation", merged_val), ("test", merged_test)]:
        formatted = model_formatter.format_for_model(data)
        path = os.path.join(output_dir, f"{pfx['formatted']}_{name}.txt")
        loader.save_text(formatted, path)
        logger.info(f"Saved formatted {name} → {path}")

    # === Process Instruction Datasets ===
    instruction_sources = config.get("datasets.instruction-sets.sources", [])
    if not instruction_sources:
        logger.info("No instruction datasets configured.")
    else:
        instruction_splits = {"train": [], "validation": [], "test": []}

        for source in instruction_sources:
            # =============== HANDLE LOCAL JSON FILES ===============
            raw_path = source["path"]
            logger.info(f"Processing local file: {raw_path}")

            # Handle difficulty merging
            if source.get("processing_type") == "with_difficulty":
                temp_path = os.path.join(output_dir, "temp_merged.json")
                raw_path = difficulty_merger.merge_difficulty_labels(
                    raw_path, source["difficulty_file"], temp_path
                )

            # Load data
            raw_data = loader.load_json_file(raw_path)
            if not raw_data:
                logger.warning(f"Skipping empty file: {raw_path}")
                continue

            # Validate structure
            if not ("instruction" in raw_data[0] and "output" in raw_data[0]):
                logger.error(f"Invalid format in {raw_path}. Skipping.")
                continue

            source_label = f"Local:{raw_path}"
            logger.info(f"Processing {len(raw_data)} records from {source_label}")

            # Clean
            cleaned = [
                {
                    "instruction": cleaner.clean_instruction_text(item.get("instruction", "")),
                    "output": cleaner.clean_instruction_text(item.get("output", ""))
                }
                for item in raw_data
            ]

            # Format and deduplicate
            formatted = instruction_formatter.format_records(cleaned)
            deduped = deduplicator.remove_duplicates(formatted, ["instruction", "output"])

            # Split
            train, val, test = splitter.split(
                deduped,
                test_size=config.get("splitting.instruction_test_size", 0.1),
                val_ratio_of_test=config.get("splitting.val_ratio_of_test", 0.48),
            )

            instruction_splits["train"].extend(train)
            instruction_splits["validation"].extend(val)
            instruction_splits["test"].extend(test)

        # Save instruction splits
        pfx = config.get("output.file_prefixes", {})
        instr_prefix = pfx.get("instruction", "instruction")
        for split_name, data in instruction_splits.items():
            if data:
                path = os.path.join(output_dir, f"{instr_prefix}_{split_name}.json")
                loader.save_json(data, path)
                logger.info(f"Saved instruction {split_name}: {len(data)} → {path}")

        # Merge with dialogue data (if configured)
        if config.get("merging.include_instruction", True):
            logger.info("Merging instruction data with dialogue data...")
            merged_train = merger.merge_lists([merged_train, instruction_splits["train"]])
            merged_val = merger.merge_lists([merged_val, instruction_splits["validation"]])
            merged_test = merger.merge_lists([merged_test, instruction_splits["test"]])

        logger.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    main()