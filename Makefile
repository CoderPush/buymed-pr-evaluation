test-accuracy:
	python main.py --path ./data/test --sample_only

test-not-in-db:
	python main.py --path ./data/not_in_db --not_in_db --threshold 0.8 --sample_only