# BuyMed Product Recognition - Evaluation

Script for evaluation of the API Service in the BuyMed Product Recognition project.

## Requirements

- Activate python venv and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Create a `.env` file in the root directory and add the following variables:

```bash
BASE_URL=<base_url>
UPLOAD_URL=<upload_url>
```

- Copy the data to the `data` folder in the root directory. The data folder should have this structure:

```bash
|- data/
    |- product_name_1/
        |- <image_1>.jpg
        |- <image_2>.jpg
        |- ...
    |- product_name_2/
        |- <image_1>.jpg
        |- <image_2>.jpg
        |- ...
```

## Usage

```bash
python main.py --path <path_to_data_folder> --threshold <threshold> [--not_in_db] [--sample_only]
```

The script can test in two modes:

- **Mode 1**: Test only on products that are in the database. This is useful for testing the performance of the API Service on found rate and accuracy.
- **Mode 2**: using the flag `--not_in_db`. Test only on products that are not in the database. This is useful for testing the performance of the API Service on new products. If this flag is set, a threshold (a number between 0 to 1) must be specified.

`--threshold` is the threshold for similarity scores. If the similarity score is below the threshold, the product is considered not found.

If the flag `--sample_only` is set, the script will randomly select one image from each product to test. If this flag is not set, the script will test on all images.

## Results

The result will be saved in "results.txt" in the root directory. The result will be in the following format:

- For mode 1:

```
Number of products: <number_of_products>
Number of images: <number_of_images>

Found rate: <found_rate>
Top 1 accuracy: <top_1_accuracy>

Average time taken per image: <time_taken>
```

- For mode 2:

```
Number of products: <number_of_products>
Number of images: <number_of_images>

Rate of not found warning: <not_found_warning_rate>

Average time taken per image: <time_taken>
```

Based on a threshold and similarity scores, a not found warning could be shown to the user. The warning rate is the rate of products that have a warning.
