from argparse import ArgumentParser
import pathlib
import pandas as pd
import requests
import os
import re
import concurrent.futures
from datetime import datetime
import time

from dotenv import load_dotenv
import unicodedata

load_dotenv()

RANDOM_SEED = 1234
BASE_URL = os.environ.get("BASE_URL")
UPLOAD_API = os.environ.get("UPLOAD_API")


def initilize_result_dict(not_in_db_flag: bool) -> dict:
    mode = 2 if not_in_db_flag else 1
    if mode == 1:
        result = {
            "Number of products": 0,
            "Number of images": 0,
            "Found rate": 0,
            "Top 1 accuracy": 0,
            "Rate of not found warning": 0,
            "Average time taken per image": 0,
            "Threshold": 0,
        }
    else:
        result = {
            "Number of products": 0,
            "Number of images": 0,
            "Rate of not found warning": 0,
            "Average time taken per image": 0,
            "Threshold": 0,
        }
    assert result["Number of images"] == 0, "Result dict must be initialized"
    return mode, result


def get_image_paths(path: str) -> dict:
    """
    Get the path of all images in the data folder
    :param path: path to the data folder
    :return: a dict with the path, product_name and sku of each image
    """
    image_paths = {"path": [], "product_name": [], "sku": []}
    raw_data_path = pathlib.Path(path)
    for path in raw_data_path.glob("**/*.jpg"):
        if path.parent.parent.name == raw_data_path.name:  # no sku available
            image_paths["path"].append(str(path))
            image_paths["product_name"].append(path.parent.name)
            image_paths["sku"].append(None)
        else:
            image_paths["path"].append(str(path))
            image_paths["product_name"].append(path.parent.parent.name)
            image_paths["sku"].append(path.parent.name)
    return image_paths


def no_accent_vietnamese(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    # s = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", s)
    # s = re.sub(r"[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]", "A", s)
    # s = re.sub(r"[èéẹẻẽêềếệểễ]", "e", s)
    # s = re.sub(r"[ÈÉẸẺẼÊỀẾỆỂỄ]", "E", s)
    # s = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", s)
    # s = re.sub(r"[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]", "O", s)
    # s = re.sub(r"[ìíịỉĩ]", "i", s)
    # s = re.sub(r"[ÌÍỊỈĨ]", "I", s)
    # s = re.sub(r"[ùúụủũưừứựửữ]", "u", s)
    # s = re.sub(r"[ƯỪỨỰỬỮÙÚỤỦŨ]", "U", s)
    # s = re.sub(r"[ỳýỵỷỹ]", "y", s)
    # s = re.sub(r"[ỲÝỴỶỸ]", "Y", s)
    # s = re.sub(r"[Đ]", "D", s)
    # s = re.sub(r"[đ]", "d", s)
    return s.strip().lower()


def sku_mapping(df_img_paths: pd.DataFrame) -> None:
    """
    Map the sku of each image to the sku in the database
    """
    # load product list from csv file
    df_product_list = pd.read_csv("product_list.csv", encoding="utf-8")
    df_product_list["ProductName"] = df_product_list["ProductName"].apply(
        no_accent_vietnamese
    )
    name_to_sku = dict(zip(df_product_list.ProductName, df_product_list.SKU))
    # preprocess product name
    df_img_paths["product_name"] = df_img_paths["product_name"].str.replace(",", "")
    df_img_paths["product_name"] = df_img_paths["product_name"].apply(
        no_accent_vietnamese
    )
    # map sku
    df_img_paths["sku"] = df_img_paths["product_name"].map(name_to_sku)
    # drop images with no sku
    df_img_paths.dropna(subset=["sku"], inplace=True)


def upload_image(image_path: str) -> str:
    files = {"file": open(image_path, "rb")}
    response = requests.post(f"{BASE_URL}{UPLOAD_API}", files=files)
    if response.ok:
        result = response.json()
        result = result["data"]

        assert len(result["similar_images"]) > 0, f"Response has no similar images"
        assert result["time_found"], f"Response has no time_found"
        assert result["similar_images"][0]["sku"], f"Similar images have no sku"
        assert result["similar_images"][0][
            "similarity"
        ], f"Similar images have no similarity score"

        return result
    else:
        raise ValueError(f"Error uploading image {image_path}")


def save_result(result):
    # Append result to result.txt with the format key: value and start with the current time
    result["Time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open("result.txt", "a") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 50)
        f.write("\n")


def evaluate_accuracy(df_img_paths: pd.DataFrame, result: dict) -> None:
    """
    Evaluate the found rate (top 3 accuracy) and top 1 accuracy of the model
    """

    def get_similar_images(index, row):
        response = upload_image(row["path"])
        similar_images = response["similar_images"]
        sku_list = [x["sku"] for x in similar_images]

        if row["sku"] in sku_list:
            result["Found rate"] += 1
            if row["sku"] == sku_list[0]:
                result["Top 1 accuracy"] += 1

        if similar_images[0]["similarity"] < result["Threshold"]:
            result["Rate of not found warning"] += 1

        # remove non-numeric characters from time_found but not . (dot)
        response["time_found"] = re.sub("[^0-9.]", "", response["time_found"])
        response["time_found"] = float(response["time_found"])
        result["Average time taken per image"] += response["time_found"]

        if (index % 50 == 0) and (index > 0):
            time_taken = time.time() - started_time
            print(f"Processed {index+1} images in {time_taken} seconds")
            print(
                f"Avg time taken per image: {result['Average time taken per image'] / (index+1)}"
            )

    started_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {
            executor.submit(get_similar_images, index, row): row
            for index, row in df_img_paths.iterrows()
        }
        for future in concurrent.futures.as_completed(tasks):
            row = tasks[future]
            try:
                future.result()
            except Exception as exc:
                print(f"ERROR in evaluate_accuracy(): Image path {row['path']}")
                print(exc)

    # For testing
    # for index, row in df_img_paths.iterrows():
    #     get_similar_images(index, row)
    #     break

    result["Found rate"] /= result["Number of images"]
    result["Top 1 accuracy"] /= result["Number of images"]
    result["Rate of not found warning"] /= result["Number of images"]
    result["Average time taken per image"] /= result["Number of images"]
    print(result)

    save_result(result)


def evaluate_not_found_rate(df_img_paths: pd.DataFrame, result: dict) -> None:
    """
    Evaluate the rate of not found warning of the model
    """

    def get_similar_images(index, row):
        response = upload_image(row["path"])
        similar_images = response["similar_images"]

        if similar_images[0]["similarity"] < result["Threshold"]:
            result["Rate of not found warning"] += 1

        response["time_found"] = re.sub("[^0-9.]", "", response["time_found"])
        response["time_found"] = float(response["time_found"])
        result["Average time taken per image"] += response["time_found"]

        if (index % 50 == 0) and (index > 0):
            time_taken = time.time() - started_time
            print(f"Processed {index+1} images in {time_taken} seconds")
            print(
                f"Avg time taken per image: {result['Average time taken per image'] / (index+1)}"
            )

    started_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {
            executor.submit(get_similar_images, index, row): row
            for index, row in df_img_paths.iterrows()
        }
        for future in concurrent.futures.as_completed(tasks):
            row = tasks[future]
            try:
                future.result()
            except Exception as exc:
                print(f"ERROR in evaluate_not_found_rate(): Image path {row['path']}")
                print(exc)

    # For testing
    # for index, row in df_img_paths.iterrows():
    #     get_similar_images(index, row)
    #     break

    result["Rate of not found warning"] /= result["Number of images"]
    result["Average time taken per image"] /= result["Number of images"]
    print(result)

    save_result(result)


def main():
    parser = ArgumentParser()
    parser.add_argument("--path", help="path to the data folder", default="./data/test")
    parser.add_argument(
        "--not_in_db",
        action="store_true",
        help="if true, will test products that are not in db",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="threshold for the similarity score",
        default=0.8,
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="if true, will sample 1 image per product",
    )
    args = parser.parse_args()

    if not args.threshold:
        raise ValueError("You need to specify a threshold for the similarity scores")
    if not (0 <= args.threshold <= 1):
        raise ValueError("The threshold must be between 0 and 1")

    # initialize result dict
    mode, result = initilize_result_dict(args.not_in_db)
    result["Threshold"] = args.threshold

    # Read data from args.path
    df_img_paths = pd.DataFrame(get_image_paths(args.path))
    print("\n" + " Loading data... ".center(50, "#"))
    print("# of images found: ", len(df_img_paths))
    print("# of products found: ", df_img_paths["product_name"].nunique())

    assert len(df_img_paths) > 0, f"No images found in the specified path {args.path}"

    # Sample images
    if args.sample_only:
        df_img_paths = df_img_paths.groupby("product_name").sample(
            n=1, random_state=RANDOM_SEED
        )
        print("# of images after sampling: ", len(df_img_paths))

    # SKU mapping
    if mode == 1:
        print("\n" + " Mapping SKU... ".center(50, "-"))
        sku_mapping(df_img_paths)
    else:
        print("-" * 50)

    df_img_paths.reset_index(drop=True, inplace=True)
    result["Number of images"] = len(df_img_paths)
    result["Number of products"] = df_img_paths["product_name"].nunique()
    print("# of testing images: ", result["Number of images"])
    print("# of testing products: ", result["Number of products"])
    assert result["Number of images"] > 0, "No images found after filtering"
    assert result["Number of products"] > 0, "No products found after filtering"

    # Evaluate
    print("\n" + " Evaluating... ".center(50, "#"))
    if mode == 1:
        print("Evaluating found rate accuracy...")
        evaluate_accuracy(df_img_paths, result)
    elif mode == 2:
        print("Evaluating not found rate...")
        evaluate_not_found_rate(df_img_paths, result)


if __name__ == "__main__":
    main()
