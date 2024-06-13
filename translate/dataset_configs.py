dataset_configs = {
    "bertaqa": {
        "dataset": "HiTZ/BertaQA",
        "dataset_configs": ["eu"],
        "dataset_split": "test",
        "dataset_fields": [
            "question",
            "candidates",
        ],
        "file_path": "../datasets/eustrivia_mt",
        "filename": "{config}.jsonl",
        "lang_codes": {
            "eu": "eus_Latn",
        },
        "lang_names": {
            "eu": "Euskara",
        },
    },
}
