import json
import os
import sys

from dotenv import load_dotenv
# from llama_index.embeddings.nvidia import NVIDIAEmbedding
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, list_collections
from sentence_transformers import SentenceTransformer

class MilvusEmbeddingManager:
    def __init__(self, host="host.docker.internal", port="19530"):
        self.host = host
        self.port = port

        load_dotenv()

        # self.nim_api_key = os.getenv("NIM_API_KEY")
        # if not self.nim_api_key:
        #     raise ValueError("API key for NIM is not set in the .env file.")
        # os.environ["NIM_API_KEY"] = self.nim_api_key

        # self.embedder = NVIDIAEmbedding(
        #     model="nvidia/nv-embedqa-e5-v5",
        #     truncate="END",
        #     api_key=self.nim_api_key
        # )
        self.embedder = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

        connections.connect("default", host=host, port=port)
        print("Connected to Milvus.")

    def create_or_load_collection(self, collection_name):
        if collection_name in list_collections():
            print(f"Collection '{collection_name}' already exists. Loading collection.")
            return Collection(name=collection_name)
        else:
            schema = CollectionSchema([
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="main_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="section_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="sub_heading_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sub_heading", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024)
            ], description=f"Embeddings collection for {collection_name}")

            print(f"Creating collection '{collection_name}'.")
            return Collection(name=collection_name, schema=schema)

    def generate_embeddings(self, text_or_image_caption):
        """Generate embeddings for the given text."""
        return self.embedder.encode(text_or_image_caption) if text_or_image_caption else [0.0] * 1024

    def process_and_insert_json(self, json_file):
        """Process JSON data from a file and insert into Milvus, handling both text and image nodes."""
        collection_name = os.path.splitext(os.path.basename(json_file))[0]
        collection = self.create_or_load_collection(collection_name)
        global_id = 1
        record_count = 0

        # Load and parse the JSON file
        with open(json_file, "r", encoding="utf-8") as file:
            try:
                json_data = json.load(file)  # Parse JSON file into a Python object
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")
                return

        def process_node(node):
            nonlocal global_id, record_count
            node_id = global_id
            global_id += 1

            # Extract metadata and content
            main_title = node.get("metadata", {}).get("main title", "")
            section_title = node.get("metadata", {}).get("section title", "")
            sub_heading = node.get("metadata", {}).get("sub heading", "").strip()
            image_path = node.get("metadata", {}).get("image", "No image available")
            
            if "image" in node.get("metadata", {}):
                content = node["metadata"]["caption"]
                # print("Content: ", content)
            else:
                content = node.get("content", "")

            main_title_emb = self.generate_embeddings(main_title)
            section_title_emb = self.generate_embeddings(section_title)
            sub_heading_emb = self.generate_embeddings(sub_heading)
            content_emb = self.generate_embeddings(content)


            # Insert text embeddings into Milvus
            collection.insert([
                [node_id],
                [main_title_emb],
                [section_title_emb],
                [sub_heading_emb],
                [content_emb],
                [content],
                [sub_heading],
                [image_path]
            ])
            record_count += 1

            # Process subheadings recursively
            for sub_node in node.get("subheadings", []):
                process_node(sub_node)

        for node in json_data:
            process_node(node)

        print(f"Data insertion complete for '{collection_name}'. Total records inserted: {record_count}.")


    def create_indexes(self, collection_name):
        """Create indexes for the collection fields."""
        collection = self.create_or_load_collection(collection_name)
        collection.flush()
        index_params = {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}

        collection.create_index("main_title_embedding", index_params)
        collection.create_index("section_title_embedding", index_params)
        collection.create_index("sub_heading_embedding", index_params)
        collection.create_index("content_embedding", index_params)
        print(f"Indexes created for '{collection_name}'.")

    def query(self, query_text, anns_field="sub_heading_embedding", limit=5, threshold=0.80):
        """Query the collection with a given text and filter results based on similarity threshold."""
        collections = list_collections()
        combined_results = {}

        print(f"Provided Answer field is: {anns_field}")

        for collection_name in collections:
            collection = self.create_or_load_collection(collection_name)
            collection.load()
            query_embedding = self.generate_embeddings(query_text)

            search_params = {"metric_type": "IP", "params": {"ef": 128}}

            if anns_field == "content_embedding":
                results = collection.search(
                    data=[query_embedding],
                    anns_field=anns_field,  # Search using content_embedding
                    param=search_params,
                    limit=limit * 2,
                    output_fields=["text", "image_path", "sub_heading"]  # Get full content & metadata
                )

                filtered_results = []
                for res in results:
                    for hit in res:
                        if hit.distance >= threshold:  # Filter based on similarity threshold
                            image_path = hit.get("image_path") or "No image provided"    # Check for image field

                            filtered_results.append({
                                "text": hit.get("text"),  # Retrieve content
                                "image": image_path,  # Assign image path or "No image provided"
                                "sub_heading": hit.entity.get("sub_heading"),
                                "collection_name": collection_name,
                                "similarity": hit.distance
                            })

            else:
                results = collection.search(
                    data=[query_embedding],
                    anns_field=anns_field,
                    param=search_params,
                    limit=limit * 2,
                    output_fields=["text", "sub_heading"]
                )
                # print("Search result: ", results)
                filtered_results = []
                for res in results:
                    for hit in res:
                        if hit.distance >= threshold:
                            filtered_results.append({
                                "text": hit.entity.get("text"),
                                "sub_heading": hit.entity.get("sub_heading"),
                                "collection_name": collection_name,
                                "similarity": hit.distance
                            })

            combined_results[collection_name] = filtered_results[:limit]

        return combined_results

    def perform_default_queries(self):
        """Perform default searches and organize results by collection and query type."""
        default_queries = ["Introduction", "Abstract", "Conclusion", "References", "Methodology", "Results"]
        collections = list_collections()
        organized_results = {query: {} for query in default_queries}

        for collection_name in collections:
            collection = self.create_or_load_collection(collection_name)
            collection.load()

            for query_text in default_queries:
                query_embedding = self.generate_embeddings(query_text)
                search_params = {"metric_type": "IP", "params": {"ef": 128}}

                results = collection.search(
                    data=[query_embedding],
                    anns_field="sub_heading_embedding",
                    param=search_params,
                    limit=1,
                    output_fields=["text", "sub_heading"]
                )

                for res in results:
                    for hit in res:
                        query_results = organized_results[query_text].setdefault(collection_name, [])
                        query_results.append({
                            "text": hit.entity.get("text"),
                            "similarity": hit.distance
                        })

        return organized_results

    def get_column_counts(self):
        """Get the count of items in each column of all collections."""
        collections = list_collections()
        column_counts = {}

        for collection_name in collections:
            collection = self.create_or_load_collection(collection_name)
            collection.load()
            collection.flush()

            fields = [field.name for field in collection.schema.fields]
            column_counts[collection_name] = {}

            for field in fields:
                results = collection.query(expr="", output_fields=[field], limit=10000)
                column_counts[collection_name][field] = len(results)

        return column_counts

# Example usage
if __name__ == "__main__":

    manager = MilvusEmbeddingManager()

    json_files = sys.argv[1:]

    if json_files:
        for json_file in json_files:
            manager.process_and_insert_json(json_file)
            manager.create_indexes(os.path.splitext(os.path.basename(json_file))[0])

    user_query = input("Enter your search query: ")

    if user_query.strip():
        content_results = manager.query(user_query, anns_field="content_embedding", limit=5)
        # print("Content Results:", json.dumps(content_results, indent=4))

        default_results = manager.perform_default_queries()
        # print("Default Results:", json.dumps(default_results, indent=4))

    column_counts = manager.get_column_counts()
    print("Column counts:", json.dumps(column_counts, indent=4))
