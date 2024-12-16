import requests
import json
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3 needed for this script
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your dynamic variables
token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MzQ0MzY3NTAsImlhdCI6MTczNDM1MDM1MCwidXNlcm5hbWUiOiJhZG1pbiJ9.dPqZcXSUO4Tj-jyE7aiuvmPpYG3TB8XgwINrB6eeTDQ"
host = "http://127.0.0.1:8443"
base_url = f"{host}/vectordb"


def generate_headers():
    return {"Authorization": f"Bearer {token}", "Content-type": "application/json"}


def create_db(name, description=None, dimension=1024):
    url = f"{base_url}/collections"
    data = {
        "name": name,
        "description": description,
        "dense_vector": {
            "enabled": True,
            "auto_create_index": False,
            "dimension": dimension,
        },
        "sparse_vector": {"enabled": False, "auto_create_index": False},
        "metadata_schema": None,
        "config": {"max_vectors": None, "replication_factor": None},
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def create_explicit_index(name):
    data = {
        "collection_name": name,
        "name": name,
        "distance_metric_type": "cosine",
        "quantization": "scalar",
        "data_type": "u8",
        "index_type": "hnsw",
        "params": {
            "num_layers": 5,
            "max_cache_size": 1000,
        }
    }
    response = requests.post(f"{base_url}/indexes", headers=generate_headers(), data=json.dumps(data), verify=False)

    return response.json()


# Function to create database (collection)
def create_db_old(vector_db_name, dimensions, max_val, min_val):
    url = f"{base_url}/collections"
    data = {
        "vector_db_name": vector_db_name,
        "dimensions": dimensions,
        "max_val": max_val,
        "min_val": min_val,
    }
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to find a database (collection) by Id
def find_collection(id):
    url = f"{base_url}/collections/{id}"

    response = requests.get(url, headers=generate_headers(), verify=False)
    return response.json()


def create_transaction(collection_name):
    url = f"{base_url}/collections/{collection_name}/transactions"
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


def create_vector_in_transaction(collection_name, transaction_id, vector):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/vectors"
    data = {"id": vector["id"], "values": vector["values"], "metadata": {}}
    print(f"Request URL: {url}")
    print(f"Request Data: {json.dumps(data)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    print(f"Response Status: {response.status_code}")
    print(f"Response Text: {response.text}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")
    return response.json() if response.text else None


def upsert_in_transaction(collection_name, transaction_id, vectors):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/upsert"
    )
    data = {"vectors": vectors}
    print(f"Request URL: {url}")
    print(f"Request Vectors Count: {len(vectors)}")
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    print(f"Response Status: {response.status_code}")
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to create vector: {response.status_code}")


def upsert_vectors_in_transaction(collection_name, transaction_id, vectors):
    url = f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/vectors"
    data = {"vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


def commit_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/commit"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    if response.status_code not in [200, 204]:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to commit transaction: {response.status_code}")
    return response.json() if response.text else None


def abort_transaction(collection_name, transaction_id):
    url = (
        f"{base_url}/collections/{collection_name}/transactions/{transaction_id}/abort"
    )
    response = requests.post(url, headers=generate_headers(), verify=False)
    return response.json()


# Function to upsert vectors
def upsert_vector(vector_db_name, vectors):
    url = f"{base_url}/upsert"
    data = {"vector_db_name": vector_db_name, "vectors": vectors}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to search vector
def ann_vector_old(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return (idd, response.json())


def ann_vector(idd, vector_db_name, vector):
    url = f"{base_url}/search"
    data = {"vector_db_name": vector_db_name, "vector": vector}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"Failed to search vector: {response.status_code}")
    result = response.json()

    # Handle empty results gracefully
    if not result.get("RespVectorKNN", {}).get("knn"):
        return (idd, {"RespVectorKNN": {"knn": []}})
    return (idd, result)


# Function to fetch vector
def fetch_vector(vector_db_name, vector_id):
    url = f"{base_url}/fetch"
    data = {"vector_db_name": vector_db_name, "vector_id": vector_id}
    response = requests.post(
        url, headers=generate_headers(), data=json.dumps(data), verify=False
    )
    return response.json()


# Function to generate a random vector with given constraints
def generate_random_vector(rows, dimensions, min_val, max_val):
    return np.random.uniform(min_val, max_val, (rows, dimensions)).tolist()


def generate_random_vector_with_id(id, length):
    values = np.random.uniform(-1, 1, length).tolist()
    return {"id": id, "values": values}


def perturb_vector(vector, perturbation_degree):
    # Generate the perturbation
    perturbation = np.random.uniform(
        -perturbation_degree, perturbation_degree, len(vector["values"])
    )
    # Apply the perturbation and clamp the values within the range of -1 to 1
    perturbed_values = np.array(vector["values"]) + perturbation
    clamped_values = np.clip(perturbed_values, -1, 1)
    vector["values"] = clamped_values.tolist()
    return vector


def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))


def magnitude(vec):
    return np.sqrt(sum(v ** 2 for v in vec))


def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    magnitude_vec1 = magnitude(vec1)
    magnitude_vec2 = magnitude(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0  # Handle the case where one or both vectors are zero vectors

    return dot_prod / (magnitude_vec1 * magnitude_vec2)


def cosine_similarity(vec1, vec2):
    # Convert inputs to numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Check if vectors have the same length
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same length")

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Check for zero vectors
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Cannot compute cosine similarity for zero vectors")

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim


def bruteforce_search(vectors, query, k=5):
    similarities = []
    for vector in vectors:
        similarity = cosine_similarity(query["values"], vector["values"])
        similarities.append((vector["id"], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def generate_vectors(txn_count, batch_count, batch_size, dimensions, perturbation_degree):
    vectors = [generate_random_vector_with_id(id, dimensions) for id in range(txn_count * batch_count * batch_size)]

    # Shuffle the vectors
    np.random.shuffle(vectors)
    return vectors


def search(vectors, vector_db_name, query):
    ann_response = ann_vector(query["id"], vector_db_name, query["values"])
    bruteforce_result = bruteforce_search(vectors, query, 5)
    return (ann_response, bruteforce_result)


def read_dataset_from_parquet():
    df1 = pd.read_parquet("test.parquet")
    df2 = pd.read_parquet("test2.parquet")

    df = pd.concat([df1, df2], ignore_index=True)
    dataset = df[['id', 'emb']].values.tolist()

    vectors = []

    for row in dataset:
        vector = {
            "id": int(row[0]),
            "values": [float(v) for v in row[1][:5]]
        }
        vectors.append(vector)

    return vectors


if __name__ == "__main__":
    # Create database
    vector_db_name = "testdb"
    dimensions = 5
    max_val = 1.0
    min_val = -1.0
    perturbation_degree = 0.25  # Degree of perturbation
    batch_size = 20
    batch_count = 100

    create_collection_response = create_db(
        name=vector_db_name,
        description="Test collection for vector database",
        dimension=dimensions,
    )
    print("Create Collection(DB) Response:", create_collection_response)
    # create_explicit_index(vector_db_name)

    vectors = read_dataset_from_parquet()

    start_time = time.time()

    for req_ct in range(1):
        transaction_id = None
        try:
            # Create a new transaction
            transaction_response = create_transaction(vector_db_name)
            transaction_id = transaction_response["transaction_id"]
            print(f"Created transaction: {transaction_id}")

            # Process vectors concurrently
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for base_idx in range(batch_count):
                    req_start = req_ct * batch_count * batch_size
                    batch_start = req_start + base_idx * batch_size
                    # upsert_in_transaction(vector_db_name, transaction_id, vectors[batch_start:batch_start+batch_size])
                    futures.append(
                        executor.submit(
                            upsert_in_transaction,
                            vector_db_name,
                            transaction_id,
                            vectors[batch_start:batch_start + batch_size]
                        )
                    )

                # Collect results
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in future: {e}")

            # Commit the transaction after all vectors are inserted
            # commit_response = commit_transaction(vector_db_name, transaction_id)
            # print(f"Committed transaction {transaction_id}: {commit_response}")
            # transaction_id = None
            # time.sleep(10)

        except Exception as e:
            print(f"Error in transaction: {e}")
            if transaction_id:
                try:
                    abort_transaction(vector_db_name, transaction_id)
                    print(f"Aborted transaction {transaction_id} due to error")
                except Exception as abort_error:
                    print(f"Error aborting transaction: {abort_error}")

    # End time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Print elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")