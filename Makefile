test-accuracy:
	python main.py --path ./data/mini_test --threshold 0.8 --sample_only

test-not-in-db:
	python main.py --path ./data/not_in_db --threshold 0.8 --not_in_db --sample_only