# Data manipulation
# --------------------------------------------------------
import pandas as pd
import numpy as np
import polars as pl
import math
import time
from collections import Counter
import re
import json


# Parallel computation inside Python
# --------------------------------------------------------
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


dict_reviewStatus_to_map = {'criteria provided,single submitter': "criteria_provided_no_conflict",
                            'no classification provided': "not_classified",
                            'criteria provided,multiple submitters,no conflicts': "criteria_provided_no_conflict",
                            'criteria provided,conflicting classifications': "criteria_provided_conflict",
                            'no assertion criteria provided': "not_classified",
                            'no classification for the single variant': "not_classified",
                            'reviewed by expert panel': "reviewed",
                            'no classifications from unflagged records': "not_classified",
                            '-': "not_classified"}

origin_dict = {"germline": "germ",
               "unknown": "unk",
               "germline,unknown": "germ",
               "not provided": "unk",
               "germline,maternal": "germmat",
               "not applicable": "unk",
               "somatic": "som",
               "germline,somatic": "germsom",
               "gemrline,paternal,unknown": "germpat",
               "de novo,germline": "germnov",
               "germline,somatic,unknown": "germsom",
               "germline,inherited,somatic": "gersominh",
               "de novo": "germnov",
               "maternal": "germmat",
               "paternal": "germpat",
               "inherited": "germinh",
               "somatic,unknown": "som",
               "de novo,germline,unknown": "germnov",
               "germline,not applicable,unknown": "germ",
               "germline,paternal,unknown": "germpat",
               "germline,inherited,maternal,unknown": "germmatinh",
               "germline,paternal": "germpat",
               "germline,paternal,uniparental": "germpat",
               "germline,tested-inconclusive": "germ",
               "de novo,unknown": "germnov",
               "de novo,germline,somatic,unknown": "germsomnov",
               "germline,maternal,unknown": "germmat",
               "germline,inherited,unknown": "germinh",
               "inherited,unknown": "germinh",
               "biparental,germline": "germbip",
               "germline,maternal,somatic": "germsommat",
               "de novo,germline,somatic": "germsomnov",
               "de novo,somatic": "germsomnov",
               "de novo,somatic,unknown": "germsomnov",
               "germline,maternal,paternal,somatic,unknown": "germsombip",
               "not applicable,somatic": "germsom",
               "germline,inherited,maternal,paternal,unknown": "germbipinh",
               "germline,not applicable,paternal,unknown": "germpat",
               "germline,inherited,maternal": "germatinh",
               "de novo,germline,not applicable": "germnov",
               "germline,not applicable,somatic": "germsom",
               "germline,maternal,not applicable,unknown": "germmat",
               "germline,inherited,not applicable": "germinh",
               "not applicable,unknown": "unk",
               "germline,not applicable,somatic,unknown": "germsom",
               "germline,not applicable": "germ",
               "maternal,not applicable": "germmat",
               "not applicable,paternal": "germpat",
               "germline,inherited,paternal,unknown": "germpatinh",
               "germline,maternal,paternal,unknown": "germbip",
               "biparental": "germbip",
               "biparental,germline,somatic,unknown": "germsombip",
               'germline,inherited,maternal,not applicable,unknown': "germmatinh",
               "germline,inherited": "germinh",
               "biparental,maternal": "germbip",
               "biparental,germline,inherited,unknown": "germbipinh",
               "biparental,germline,unknown": "germbip",
               "germline,inherited,somatic,unknown": "germsominh",
               "paternal,unknown": "germpat",
               "germline,maternal,somatic,unknown": "germsommat",
               "germline,paternal,somatic,unknown": "germsompat",
               "germline,uniparental": "germ",
               "de novo,germline,inherited,maternal": "germmatnovinh",
               "de novo,germline,paternal": "germpatnov",
               "de novo,germline,maternal": "germmatnov",
               "de novo,germline,maternal,unknown": "germmatnov",
               "germline,maternal,paternal": "germbip",
               "de novo,germline,inherited,unknown": "germnovinh",
               "maternal,unknown": "germmat",
               "inherited,not applicable": "germinh",
               "germline,inherited,paternal": "germpatinh",
               "de novo,germline,maternal,somatic,unknown": "germsommatnov",
               "de novo,germline,paternal,somatic,unknown": "germsompatnov",
               "germline,tested-inconclusive,unknown": "germ",
               "germline,paternal,somatic": "gemrsompat",
               "de novo,germline,inherited,somatic,unknown": "germsomnov",
               "germline,not-reported,unknown": "germ",
               "de novo,germline,paternal,unknown": "germpatnov",
               "germline,uniparental,unknown": "germ",
               "not applicable,somatic,unknown": "germsom",
               "germline,somatic,tested-inconclusive,unknown": "germsom",
               "de novo,germline,inherited,paternal,somatic,unknown": "germsompatnovinh"
              }

# Function to clean, simplify, remove numbers and specified symbols, sort, deduplicate, and handle empty results
def simplify_json(json_str):
    mutations = json.loads(json_str)
    cleaned = set()
    for item in mutations:
        mutation = item[0]
        simplified = re.sub(r'[a-z]\.\-?\d+', '', mutation)  # Remove "g.<number>" and "c.<number>"
        simplified = re.sub(r'\d+', '', simplified)  # Remove standalone numbers
        simplified = re.sub(r'[+\[\]_*\.\=]', '', simplified)  # Remove specified symbols
        if simplified != '-':
            cleaned.add(simplified)
    result = ', '.join(sorted(cleaned))
    return result if result else 'no_info'

# Define a function to clean the HTML code
def clean_html(html):

    match = re.search(r'<small>based on: (.*?)</small>', html)
    return match.group(1) if match else 'no info'


def cleaning(df: pd.DataFrame, df_name):

    if df_name in ["cancermama_clinvarmain", "variation_information"]:
        # Change pandas to polars
        df = pl.from_pandas(df)

        # Calculate the threshold for 50% missing values
        threshold: float = len(df) * 0.5

        # Identify columns with more than 50% missing values
        columns_to_remove: List[str] = [
            col for col in df.columns if df[col].null_count() > threshold]

        # Identify columns with only a single unique value
        columns_to_remove.extend(
            [col for col in df.columns if df[col].n_unique() == 1])
        columns_to_remove.extend(
            [col for col in df.columns if col == "lastEval"])

        # Remove the identified columns
        df: pl.DataFrame = df.drop(columns_to_remove)

        # Isolate the specific part of the string in the origName column and convert it to lowercase
        if "origName" in df.columns:

            df = df.with_columns(
                pl.col("origName")
                # Extract the specific part
                .str.extract(r'(\w+\.\w+\(.*?\):.*?$)', 1)
                    .str.to_lowercase()
                    .alias("ClinInfo")  # Save it in a new column
            )

        if "ClinInfo" in df.columns:
            df = df.with_columns(
                pl.col("ClinInfo")
                .str.to_lowercase()
            )

        if "reviewStatus" in df.columns:
            df = df.with_columns(pl.col("reviewStatus")
                                 .map_elements(clean_html, return_dtype=str)
                                 .map_elements(lambda x: dict_reviewStatus_to_map
                                 .get(x, x), return_dtype=str)
                                 .alias("reviewStatus"))

        if "_jsonHgvsTable" in df.columns:
            df = df.with_columns(pl.col("_jsonHgvsTable")
                     .map_elements(simplify_json, return_dtype=str)
                     .alias("simplified_hgvs"))
            df = df.with_columns(pl.col("simplified_hgvs")
                   .str.strip_chars()
                   .str.replace("---", "-")
                   .str.replace("--", "-"))

        # Change to pandas
        df = df.to_pandas()
        # Count occurrences of each category in simplified_hgvs
        if "simplified_hgvs" in df.columns:
            df["simplified_hgvs"] = df["simplified_hgvs"].str.replace("--", "-")
            category_counts = df["simplified_hgvs"].value_counts()
            # Replace categories with count of 2 or less with "other"
            df["simplified_hgvs"] = df["simplified_hgvs"].apply(lambda x: "other" if category_counts[x] < 2 else x)
            df = df.drop(["_jsonHgvsTable"], axis=1)
            
        if "origin" in df.columns:
            df["origin"] = df["origin"].replace(origin_dict)
            category_counts = df["origin"].value_counts()
            # Replace categories with count of 2 or less with "other"
            df["origin"] = df["origin"].apply(lambda x: "other" if category_counts[x] <= 3 else x)
            
        else:
            pass
            
        object_cols = df.select_dtypes(include=['object']).columns
        for i in object_cols:
            df[i] = df[i].str.lower()
            df[i] = df[i].astype("category")
        return df

    if df_name == "UP.geneVsrepList":
        df = parse_dataframe(df)
        return df

    else:
        raise ValueError(
            "invalid df name, valid options are cancermama_clinvarmain, variation_information and UP.geneVsrepList")


def compare_columns(df: pd.DataFrame, col1_name: str, col2_name: str) -> str:

    # print(f"{col1_name} vs {col2_name}")
    differences = df[col1_name].compare(df[col2_name])
    if differences.empty:
        return col2_name
    elif len(differences) <= 10:
        print(
            f"Columns '{col1_name}' and '{col2_name}' have {len(differences)} differences.")
    return None


def compare_and_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    columns_to_drop = []
    columns = df.columns

    object_cols = df.select_dtypes(include=['category']).columns

    for i in object_cols:
        df[i] = df[i].astype("object")

    with ThreadPoolExecutor() as executor:
        futures = []

        for i, col1 in enumerate(columns):

            for col2 in columns[i + 1:]:
                futures.append(executor.submit(
                    compare_columns, df, col1, col2))

        for future in as_completed(futures):
            result = future.result()
            if result:
                columns_to_drop.append(result)

    df = df.drop(columns=columns_to_drop)

    return df

# Function to process the DataFrame


def parse_dataframe(df):

    df = df.rename(columns={4: "gene", 6: "sign", 7: "to_parse"})
    cols_to_drop = [0, 1, 2, 3, 5]
    result = []

    for i in df.columns:

        if i in cols_to_drop:
            df.drop([i], axis=1, inplace=True)
        else:
            pass

    for index, row in df.iterrows():
        gene = row['gene']
        # Split and remove the first empty element
        to_parse = row['to_parse'].split('>')[1:]

        # Extract kind and position
        parsed_data = [tuple(item.split('|')) for item in to_parse]

        # Count occurrences of each (kind, position) pair
        counts = Counter(parsed_data)

        # Append to result list
        for (kind, position), count in counts.items():
            result.append({'gene': gene, 'kind': kind,
                          'position': position, 'counts': count})

    return pd.DataFrame(result)
