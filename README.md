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

### The procedure of the script

In the first step, the script will find all images in the data folder.

Then it uses `product_list.csv` to map the product name with the SKU which is used in the ML Service. The `product_list.csv` file should have the following columns:

```csv
ProductName,ImageUrl,SKU
```

Depend on the mode, the script will select the products to test. If the mode is 1, the script will select the products that are in the database. If the mode is 2, the script will select the products that are not in the database.

Then it will send the image to the API Service. The API Service will return the top 3 suggestions for the image.

## Results

The overall result will be saved in "result.txt" in the root directory. The result will be in the following format:

- For mode 1:

```
Number of products: <number_of_products>
Number of images: <number_of_images>
Found rate: <found_rate>
Top 1 accuracy: <top_1_accuracy>
Rate of not found warning: <not_found_warning_rate>
Threshold: <threshold>
Average time taken per image: <time_taken>
Time: <time>
```

- For mode 2:

```
Number of products: <number_of_products>
Number of images: <number_of_images>
Rate of not found warning: <not_found_warning_rate>
Threshold: <threshold>
Average time taken per image: <time_taken>
Time: <time>
```

Based on a threshold and similarity scores, a not found warning could be shown to the user. The warning rate is the rate of products that have a warning.

A detail result will be saved in "detail*result*<mode>\_<timestamp>.csv" in the root directory. The csv file contains the following columns:

- `path`: path to the image
- `sku`: sku of the product
- `is_in_db`: whether the product is in the database
- `1st_suggestion`: sku of the 1st suggestion
- `1st_similarity`: similarity score of the 1st suggestion
- `2nd_suggestion`: sku of the 2nd suggestion
- `2nd_similarity`: similarity score of the 2nd suggestion
- `3rd_suggestion`: sku of the 3rd suggestion
- `3rd_similarity`: similarity score of the 3rd suggestion
- `top_1`: whether the 1st suggestion is correct
- `found`: whether the product is found within the 3 suggestions
